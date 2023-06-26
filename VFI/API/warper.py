import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

# 와핑 함수 : double tensor [B, C, H, W]
def warp(img, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, C, H, W] flow
    """
    B, C, H, W = img.size()

    # mesh grid
    xx = torch.arange(0, W).permute(1,-1).repeat(H,1)  # [H, W]
    yy = torch.arange(0, H).permute(-1,1).repeat(1,W)  # [W, H]
    xx = xx.permute(1,1,H,W).repeat(B,1,1,1)           # [1, 1, H, W]
    yy = yy.permute(1,1,H,W).repeat(B,1,1,1)           # [1, 1, H, W]
    grid = torch.cat((xx,yy),1).float()             # [1, 2, H, W]

    # cuda setting & torch.aurograd Variable 변수 생성.
    if img.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo    # [1, 2, H, W]

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)  # [1, H, W, 2]
    output = nn.functional.grid_sample(img, vgrid)  # [1, 3, 320, 320]
    mask = torch.autograd.Variable(torch.ones(img.size())).cuda()
    mask = nn.functional.grid_sample(mask.type(torch.Tensor), vgrid)
        
    mask[mask<0.999] = 0
    mask[mask>0] = 1
        
    return output*mask

class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...

    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """

    # params = 너비 / 높이 / Cuda 디바이스
    def __init__(self, W, H, device):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """

        super(backWarp, self).__init__()
        # Optical Flow를 반환하기 위한 그리드 세팅.
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))      # [[0, 1, 2, ... , W-1] * H], [[W, W+1, W+2, ... , 2W-1] * H], 
        self.W = W
        self.H = H
        self.gridX = torch.tensor(gridX, requires_grad=False, device=device)    # (520, 960)
        self.gridY = torch.tensor(gridY, requires_grad=False, device=device)    # (520, 960)

    def make_mask(self, u, v):
        def calculate_distance(x, y):
            distance = math.sqrt(x**2 + y**2)
            return distance
        mask = torch.ones_like(u.squeeze())
        u = torch.flatten(u)
        v = torch.flatten(v)
        amp = [calculate_distance(x,y) for x,y in zip(u,v)]
        amp_2d = np.array(amp).reshape(520,960)
        mask[amp_2d>=50]=0
        print(mask.shape)
        return mask
    
    # I1과 F01을 통해 I0 이미지 반환.
    # I0 = backwarp(I1, F_0_1)
    # params = I1, F01
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)

        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.

        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]            # x 변화량  tensor([1, 520, 960])
        v = flow[:, 1, :, :]            # y 변화량  tensor([1, 520, 960])
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u        # tensor([1, 520, 960])
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v        # tensor([1, 520, 960])

        # Normalize Flow : (range -1 to 1)
        x = 2*(x/self.W - 0.5)          # tensor([1, 520, 960])
        y = 2*(y/self.H - 0.5)          # tensor([1, 520, 960])

        # stacking X and Y
        grid = torch.stack((x,y), dim=3)    # # tensor([1, 520, 960, 2])

        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid)     # [(N, C, H, W), (N, H, W, 2)]  # tensor([1, 3, 520, 960])
        # mask = self.make_mask(u, v)
        # imgOut = img*mask
        return imgOut   # (N, C, H, W)