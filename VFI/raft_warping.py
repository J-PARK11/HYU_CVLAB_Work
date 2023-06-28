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

from API import warper, utils, depth, softsplat
from dataloader.RAFT_dataloder import get_loader

class raft:

    def __init__(self, args, device):
        super(raft, self).__init__()
        self.args = args
        self.device = device
        self._build_model()
        H, W = self.args.crop_size
        self.back_warp = warper.backWarp(W, H, self.device).to(self.device)

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
            # self.ft0gt = self.model(gt_img, input_img[0])[-1]           
            # self.ft1gt = self.model(gt_img, input_img[1])[-1]           

            # Depth Estimation
            self.d0 = depth.midas_pred(self.midas, input_img[0]).unsqueeze(1)        # [4,1,512,960]
            self.d1 = depth.midas_pred(self.midas, input_img[1]).unsqueeze(1)
                
            # Flow & Depth Interpolation
            self.ft0 = self.interpolate_by_flow(self.f10*0.5, self.f10*0.5)           
            self.ft1 = self.interpolate_by_flow(self.f01*0.5, self.f01*0.5)

            self.d0t = self.interpolate_by_flow(self.f01*0.5, self.d0)
            self.d1t = self.interpolate_by_flow(self.f10*0.5, self.d1)
                
            # BackWarping
            # self.git0gt = self.back_warp(input_img[0], self.ft0gt)      # [4,3,512,960]
            # self.git1gt = self.back_warp(input_img[1], self.ft1gt)
            self.git0 = self.back_warp(input_img[0], self.ft0)
            self.git1 = self.back_warp(input_img[1], self.ft1)

            # Hole Imputation
            # self.git0gt = self.interpolate_hole(self.git0gt, self.git1gt)
            # self.git1gt = self.interpolate_hole(self.git1gt, self.git0gt)
            self.git0 = self.interpolate_hole(self.git0, self.git1)
            self.git1 = self.interpolate_hole(self.git1, self.git0)

            # Synthesis
            # self.syn_i_gt = self.depth_guide_synthesis(self.git0gt, self.git1gt, self.d0t, self.d1t)                  # [4,3,512,960]
            self.syn_i = self.depth_guide_synthesis(self.git0, self.git1, self.d0t, self.d1t)

        return self.syn_i
    
    def interpolate_hole(self, image, other_image):
        mask = torch.zeros_like(image)
        condition = (image == 0)
        mask[condition] = 1
        new = mask * other_image
        image = image + new
        return image

    def interpolate_by_flow(self, flow, target):    # Replace to Softsplat
        batch, channels, height, width = target.size()
        itp_flow = torch.zeros_like(target)

        grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height))
        grid_x = (grid_x.float().T).to(self.device)
        grid_y = (grid_y.float().T).to(self.device)

        flow_x = flow[:, 0]
        flow_y = flow[:, 1]

        new_x = grid_x.unsqueeze(0).expand(batch, -1, -1) + flow_x
        new_y = grid_y.unsqueeze(0).expand(batch, -1, -1) + flow_y

        valid_mask = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)

        new_x = new_x.clamp(0, width - 1)
        new_y = new_y.clamp(0, height - 1)

        for b in range(batch):
            for c in range(channels):
                itp_flow[b, c][new_y[b, valid_mask[b]].long(), new_x[b, valid_mask[b]].long()] \
                    = target[b, c, valid_mask[b]]

        return itp_flow

    def depth_guide_synthesis(self, gi0t, gi1t, d0t, d1t):
        mask = torch.zeros_like(gi0t)
        condition = (d0t > d1t).repeat(1,3,1,1)
        mask[condition] = 1
        valid_i0 = mask * gi0t
        valid_i1 = (1 - mask) * gi1t
        syn_i = valid_i0 + valid_i1
        result = syn_i.view((-1, 3, 512, 960))
        return result

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    print('\n>>>>>>>>>>>> RAFT Flow based Warping Start <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')
    exe = raft(args)
    print('\n>>>>>>>>>>>>>>>>> Main Implementation <<<<<<<<<<<<<<<<<<<<')
    exe.test()
    print('\n>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')



