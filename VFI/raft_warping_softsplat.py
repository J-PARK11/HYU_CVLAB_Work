import os
import os.path as osp
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

import torch
from tqdm import tqdm
import torchvision
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings('ignore')

from API import utils, depth, softsplat
from dataloader.RAFT_dataloder import get_loader

class raft:

    def __init__(self, args, device):
        super(raft, self).__init__()
        self.args = args
        self.device = device
        self.out_root = self.args.out_root
        self._build_model()
        H, W = self.args.crop_size

    def _build_model(self):
        # Raft Flow Model & Weight load
        if self.args.flow_model == 'raft_small':
            self.weight = torchvision.models.optical_flow.Raft_Small_Weights.C_T_V2
            self.model = torchvision.models.optical_flow.raft_small(self.weight).to(self.device)
        elif self.args.flow_model == 'raft_large':
            self.weight = torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2
            self.model = torchvision.models.optical_flow.raft_large(self.weight).to(self.device)
        print(f"{self.args.flow_model} #params", sum([p.numel() for p in self.model.parameters()]))

        # MiDas Mono-Depth Model & Transformer Load
        self.midas, self.midas_transform = depth.Midas_depth(self.args.depth_model)
        self.midas.to(self.device)
        print(f"{self.args.depth_model} #params", sum([p.numel() for p in self.midas.parameters()]))

    # ---------------------------- train & test ---------------------------- #
    def exe(self, input_img, gt_img, input_path, gt_path):
        
        self.model.eval()
        with torch.no_grad():
            
            # Flow Estimation
            self.f01 = self.model(input_img[0], input_img[1])[-1]       # [4,2,512,960]
            self.f10 = self.model(input_img[1], input_img[0])[-1]       

            # Softslpat Flow Warping for Visualization
            self.f0t = softsplat.FunctionSoftsplat(tenInput=self.f01*0.5, tenFlow=self.f01*0.5,     # [4,2,512,960]
                                                    tenMetric=None, strType='average')   
            self.f1t = softsplat.FunctionSoftsplat(tenInput=self.f10*0.5, tenFlow=self.f10*0.5,     
                                                    tenMetric=None, strType='average') 

            # Depth Estimation
            self.d0 = depth.midas_pred(self.midas, input_img[0]).unsqueeze(1)        # [4,1,512,960]
            self.d1 = depth.midas_pred(self.midas, input_img[1]).unsqueeze(1)
                
            # Softsplat Frame & Depth Warping
            self.gi0t = softsplat.FunctionSoftsplat(tenInput=input_img[0], tenFlow=self.f01*0.5,     # [4,3,512,960]
                                                    tenMetric=None, strType='average')
            self.gi1t = softsplat.FunctionSoftsplat(tenInput=input_img[1], tenFlow=self.f10*0.5,
                                                    tenMetric=None, strType='average')

            self.d0t = softsplat.FunctionSoftsplat(tenInput=self.d0, tenFlow=self.f01*0.5,          # [4,1,512,960]
                                                    tenMetric=None, strType='average')
            self.d1t = softsplat.FunctionSoftsplat(tenInput=self.d1, tenFlow=self.f10*0.5,
                                                    tenMetric=None, strType='average')
                
            # Hole Imputation
            self.gi0t = self.interpolate_hole(self.gi0t, self.gi1t)                       # [4,3,512,960]
            self.gi1t = self.interpolate_hole(self.gi1t, self.gi0t)

            # Synthesis
            self.syn_i = self.depth_guide_synthesis(self.gi0t, self.gi1t, self.d0t, self.d1t)       # [4,3,512,960]


            # Visualzie & Save file
            # self.vis_flow(input_path, gt_path, save=True)               
            # self.vis_warp(input_path, gt_path, save=True)
            # self.vis_depth(input_path, gt_path, save=True)
    
        # Result Print
        print(f'Last Sample: {gt_path[0].split("/")[-2]}')
        print(f'f0t Flow Estimated: {self.f0t.shape}, {self.f0t.mean()}, {self.f0t.dtype}')
        print(f'd0t Depth Estimated: {self.d0t.shape}, {self.d0t.mean()}, {self.d0t.dtype}')
        print(f'gi0t Warped: {self.gi0t.shape}, {self.gi0t.mean()}, {self.gi0t.dtype}')
        print(f'syn_i Warped: {self.syn_i.shape}, {self.syn_i.mean()}, {self.syn_i.dtype}')

        return self.syn_i
    
    def interpolate_hole(self, image, other_image):
        mask = torch.zeros_like(image)
        condition = (image == 0)
        mask[condition] = 1
        new = mask * other_image
        image = image + new
        return image

    def depth_guide_synthesis(self, gi0t, gi1t, d0t, d1t):
        mask = torch.zeros_like(gi0t)
        condition = (d0t > d1t).repeat(1,3,1,1)
        mask[condition] = 1
        valid_i0 = mask * gi0t
        valid_i1 = (1 - mask) * gi1t
        syn_i = valid_i0 + valid_i1
        result = syn_i.view((-1, 3, 512, 960))
        return result

    def vis_flow(self, input_path, gt_path, save=True):
        
        for i, name in enumerate(gt_path):
            
            # Save format
            save_f01 = self.f01.cpu().detach()[i]
            save_f10 = self.f10.cpu().detach()[i]
            save_f0t = self.f0t.cpu().detach()[i]
            save_f1t = self.f1t.cpu().detach()[i]
            save_f0tgt = self.f0tgt.cpu().detach()[i]
            save_f1tgt = self.f1tgt.cpu().detach()[i]

            # Numpy
            name = name.split('/')[-2]
            np.save(self.out_root + name + '/f01.npy', save_f01)
            np.save(self.out_root + name + '/f10.npy', save_f10)
            np.save(self.out_root + name + '/f0t.npy', save_f0t)
            np.save(self.out_root + name + '/f1t.npy', save_f1t)
            np.save(self.out_root + name + '/f0tgt.npy', save_f0tgt)
            np.save(self.out_root + name + '/f1tgt.npy', save_f1tgt)

            # Flow
            utils.save_flow(save_f01, self.out_root + name + '/f01.png')
            utils.save_flow(save_f10, self.out_root + name + '/f10.png')
            utils.save_flow(save_f0t, self.out_root + name + '/f0t.png')
            utils.save_flow(save_f1t, self.out_root + name + '/f1t.png')
            utils.save_flow(save_f0tgt, self.out_root + name + '/f0tgt.png')
            utils.save_flow(save_f1tgt, self.out_root + name + '/f1tgt.png')
    
    def vis_warp(self, input_path, gt_path, save=True):
        
        for i, name in enumerate(gt_path):

            # Save format    
            save_gi0t = self.gi0t.cpu().detach()[i].permute(1,2,0)
            save_gi1t = self.gi1t.cpu().detach()[i].permute(1,2,0)
            save_gi0tgt = self.gi0tgt.cpu().detach()[i].permute(1,2,0)
            save_gi1tgt = self.gi1tgt.cpu().detach()[i].permute(1,2,0)
            save_syn_i_gt = self.syn_i_gt.cpu().detach()[i].permute(1,2,0)
            save_syn_i = self.syn_i.cpu().detach()[i].permute(1,2,0)

            # Warp
            name = name.split('/')[-2]
            utils.save_img(save_gi0t, self.out_root + name +'/gi0t.png')
            utils.save_img(save_gi1t, self.out_root + name +'/gi1t.png')
            utils.save_img(save_gi0tgt, self.out_root + name + '/gi0tgt.png')
            utils.save_img(save_gi1tgt, self.out_root + name +'/gi1tgt.png')

            # Synthesis
            utils.save_img(save_syn_i_gt, self.out_root + name +'/syn_i_gt.png')
            utils.save_img(save_syn_i, self.out_root + name +'/syn_i.png')

    def vis_depth(self, input_path, gt_path, save=True):
        
        for i, name in enumerate(gt_path):

            # Save format    
            save_d0 = self.d0.cpu().detach()[i,0]
            save_d1 = self.d1.cpu().detach()[i,0]
            save_d0t = self.d0t.cpu().detach()[i,0]
            save_d1t = self.d1t.cpu().detach()[i,0]
            save_d0tgt = self.d0tgt.cpu().detach()[i,0]
            save_d1tgt = self.d1tgt.cpu().detach()[i,0]

            # Depth
            name = name.split('/')[-2]
            utils.save_img(save_d0, self.out_root + name +'/d0.png')
            utils.save_img(save_d1, self.out_root + name +'/d1.png')
            utils.save_img(save_d0t, self.out_root + name +'/d0t.png')
            utils.save_img(save_d1t, self.out_root + name +'/d1t.png')
            utils.save_img(save_d0tgt, self.out_root + name +'/d0tgt.png')
            utils.save_img(save_d1tgt, self.out_root + name +'/d1tgt.png')


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    print('\n>>>>>>>>>>>> RAFT Flow based Warping Start <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')
    exe = raft(args)
    print('\n>>>>>>>>>>>>>>>>> Main Implementation <<<<<<<<<<<<<<<<<<<<')
    
    print('\n>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')



