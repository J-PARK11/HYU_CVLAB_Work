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

from API import warper
from API import utils
from API import depth

def create_parser():
    parser = argparse.ArgumentParser(description='Raft Kernel')
    
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    

    parser.add_argument('--img0', default='./data/shaman1/frame_0022.png', type=str)
    parser.add_argument('--gt', default='./data/shaman1/frame_0023.png', type=str)
    parser.add_argument('--img1', default='./data/shaman1/frame_0024.png', type=str)

    # parser.add_argument('--img0', default='./data/market1/frame_0013.png', type=str)
    # parser.add_argument('--gt', default='./data/market1/frame_0014.png', type=str)
    # parser.add_argument('--img1', default='./data/market1/frame_0015.png', type=str)

    # parser.add_argument('--img0', default='./data/synthetic/frame_0014.png', type=str)
    # parser.add_argument('--gt', default='./data/synthetic/frame_0015.png', type=str)
    # parser.add_argument('--img1', default='./data/synthetic/frame_0016.png', type=str)

    # parser.add_argument('--img0', default='./data/images/0104.jpg', type=str)
    # parser.add_argument('--gt', default='./data/images/0105.jpg', type=str)
    # parser.add_argument('--img1', default='./data/images/0106.jpg', type=str)

    parser.add_argument('--out_dir', default='./output/shaman1/', type=str)
    
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--num_workers', default=4, type=int)

    # model parameters
    parser.add_argument('--model', default='raft_large', choices=['raft_large', 'raft_small'])
    parser.add_argument('--depth_type', default='DPT_Large', choices=['DPT_Large', 'DPT_Hybrid', 'MiDaS_small'])
    

    return parser

class raft:

    def __init__(self, args):
        super(raft, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.out_dir = self.args.out_dir

        self.device = self._acquire_device()
        self._get_data()
        self._build_model()

    def _acquire_device(self):
        if self.args.use_gpu:
            # args.use_gpu가 True일 경우, CUDA에 args.gpu에 기재된 GPU Number를 입력.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
        else:
            device = torch.device('cpu')
        return device   

    def _build_model(self):
        # Model & Weight load
        if self.args.model == 'raft_small':
            self.weight = torchvision.models.optical_flow.Raft_Small_Weights.C_T_V2
            self.model = torchvision.models.optical_flow.raft_small(self.weight).to(self.device)
        elif self.args.model == 'raft_large':
            self.weight = torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2
            self.model = torchvision.models.optical_flow.raft_large(self.weight).to(self.device)

        # Monocular Depth Load
        # self.midas, self.midas_transform = depth.Midas_depth(args.depth_type)
        # self.midas.to(self.device)

    def _get_data(self):
        crop_size = [520,960]
        self.img0 = torch.tensor(utils.load_image([self.args.img0]).astype('float32')).swapaxes(1,3).swapaxes(2,3)
        self.img1 = torch.tensor(utils.load_image([self.args.img1]).astype('float32')).swapaxes(1,3).swapaxes(2,3)
        self.gt = torch.tensor(utils.load_image([self.args.gt]).astype('float32')).swapaxes(1,3).swapaxes(2,3)
        if (self.img0.shape[-1] > 1000) or (self.img0.shape[-1] % 8 != 0) or (self.img0.shape[-2] % 8 != 0):
            self.img0 = F.resize(self.img0, size=crop_size, antialias=False)
            self.img1 = F.resize(self.img1, size=crop_size, antialias=False)
            self.gt = F.resize(self.gt, size=crop_size, antialias=False)

    # ---------------------------- train & test ---------------------------- #
    def test(self):
        B,C,H,W = self.img0.shape

        # Flow
        self.flowt0 = self.model(self.gt.to(self.device), self.img0.to(self.device))[-1]
        self.flowt1 = self.model(self.gt.to(self.device), self.img1.to(self.device))[-1]
        print(f'flow shape: {self.flowt0.shape}')
        print('min:',self.flowt0.min(), 'max:', self.flowt0.max(), 'mean:',self.flowt0.mean())

        # Depth
        # print(self.img0.shape)
        # print(self.img0[0].permute(1,2,0).shape)
        # self.depth0 = self.midas_transform(self.img0[0].permute(1,2,0).numpy()).to(self.device)
        # self.depth1 = self.midas_transform(self.img1[0].permute(1,2,0).numpy()).to(self.device)
        # print(self.depth0.shape)
        # self.depth0 = depth.midas_pred(self.midas, self.depth0, self.img0)
        # self.depth1 = depth.midas_pred(self.midas, self.depth1, self.img1)
        # print(self.depth0.shape)
        # save_depth0 = self.depth0.cpu().detach().numpy()
        # utils.save_img(save_depth0, self.out_dir + 'depth0.png')

        # flot1 = utils.read_flo('./data/synthetic/frame_0015.flo')
        # flot1 = F.resize(torch.tensor(flot1).permute(2,0,1), size=[520,960], antialias=False).unsqueeze(0).to(self.device)
        # print(flot1.shape)
        
        # Warping
        with torch.set_grad_enabled(False):
            back_warp = warper.backWarp(W, H, self.device).to(self.device)
            git0 = back_warp(self.img0.to(self.device), self.flowt0)
            git1 = back_warp(self.img1.to(self.device), self.flowt1)
            # git1gt = back_warp(self.img1.to(self.device), flot1)
            print(f'warped: {git0.shape}')

        # Synthesis
        # mask_git0 = self.create_mask(git0)
        # mask_git1 = self.create_mask(git1)
        # git0 = self.apply_mask(git0, mask_git0, git1)
        # git1 = self.apply_mask(git1, mask_git1, git0)
        git0 = self.interpolate_hole(git0,git1)
        git1 = self.interpolate_hole(git1,git0)
        syn_i = (git0 + git1) / 2

        # Save File
        save_flowt0 = self.flowt0.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)
        save_flowt1 = self.flowt1.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)
        save_git0 = git0.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)
        save_git1 = git1.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)
        save_syn_i = syn_i.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)

        # save_flowt1gt = flot1.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)
        # save_git1gt = git1gt.cpu().detach().numpy()[0].swapaxes(0,2).swapaxes(0,1)

        # np.save(self.out_dir + 'flowt0.npy', save_flowt0)
        # np.save(self.out_dir + 'flowt1.npy', save_flowt1)
        utils.save_flow(self.flowt0[0], self.out_dir + 'ft0.png')
        utils.save_flow(self.flowt1[0], self.out_dir + 'ft1.png')
        utils.save_img(save_git0, self.out_dir + 'git0.png')
        utils.save_img(save_git1, self.out_dir + 'git1.png')
        utils.save_img(save_syn_i, self.out_dir + 'syn_i.png')

        # utils.save_img(save_git1gt, self.out_dir + 'git1gt.png')
        # utils.save_flow(flot1[0], self.out_dir + 'ft1gt.png')
    
    def interpolate_hole(self, image, other_image):
        mask = torch.zeros_like(image)
        condition = (image == 0)
        mask[condition] = 1
        new = mask * other_image
        image = image + new
        return image

if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>> Start <<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print('Initialize')
    exp = raft(args)
    print('Test')
    exp.test()
    print('>>>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
