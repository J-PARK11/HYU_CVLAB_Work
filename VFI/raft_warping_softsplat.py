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
        print(f"{self.args.flow_model} softsplat average #params", sum([p.numel() for p in self.model.parameters()]))

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

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    print('\n>>>>>>>>>>>> RAFT Flow based Warping Start <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')
    exe = raft(args)
    print('\n>>>>>>>>>>>>>>>>> Main Implementation <<<<<<<<<<<<<<<<<<<<')
    
    print('\n>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')



