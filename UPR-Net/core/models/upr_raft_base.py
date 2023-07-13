import os
import torch
import math
import numpy
import torch.nn.functional as F
import torch.nn as nn

from ..utils import correlation
from ..models.softsplat import softsplat
from ..models.raft import raft

#**************************************************************************************************#
# => Feature Pyramid
#**************************************************************************************************#
class FeatPyramid(nn.Module): 
    """
    Feature Encoder:
    A 3-stage feature pyramid, which by default is shared by the motion
    estimator and synthesis network.
    ==> 4 Conv per 3 stages & Downsample at first layer of stage. (1 -> 1/4)
    ### Channel Tracking ###
    3 -> 16 -> 32 -> 64 ->
    """
    def __init__(self):
        super(FeatPyramid, self).__init__()
        self.conv_stage0 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage1 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
        self.conv_stage2 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                    stride=2, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))

    def forward(self, img):
        C0 = self.conv_stage0(img)
        C1 = self.conv_stage1(C0)
        C2 = self.conv_stage2(C1)
        return [C0, C1, C2]



#**************************************************************************************************#
# => Frame Synthesis
#**************************************************************************************************#
class SynthesisNetwork(nn.Module):
    """
    Frame Synthesis Module
    ### Channel Tracking ###
    Input Channel = last itp image (3) + Warped img (3 * 2) + bi-directional flow(4) + input frame (3 * 2) = 19
    Pred 5 Channel => residual of itp image (3), Occlusion Map (2)
    ==> 3 Conv per 3 stages & Downsample at first layer of stage. (1 -> 1/4)
    ==> 3 Conv per 3 stages & Upsample at first layer of stage. (1/4 -> 1)
    """
    def __init__(self):
        super(SynthesisNetwork, self).__init__()
        
        input_channels = 9+4+6
        self.encoder_conv = nn.Sequential(
                nn.Conv2d(in_channels=input_channels, out_channels=32,
                    kernel_size=3, stride=1, padding=1),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.encoder_down1 = nn.Sequential(
                nn.Conv2d(in_channels=32 + 16 + 16, out_channels=64,
                    kernel_size=3, stride=2, padding=1),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64))
        self.encoder_down2 = nn.Sequential(
                nn.Conv2d(in_channels=64 + 32 + 32, out_channels=128,
                    kernel_size=3, stride=2, padding=1),
                nn.PReLU(num_parameters=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=128),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=128))
        self.decoder_up1 = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=128 + 64 + 64,
                    out_channels=64, kernel_size=4, stride=2,
                    padding=1, bias=True),
                nn.PReLU(num_parameters=64),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=64))
        self.decoder_up2 = nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=64 + 64,
                    out_channels=32, kernel_size=4, stride=2,
                    padding=1, bias=True),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.decoder_conv = nn.Sequential(
                nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                    stride=1, padding=1),
                nn.PReLU(num_parameters=32))
        self.pred = nn.Conv2d(in_channels=32, out_channels=5, kernel_size=3,
                stride=1, padding=1)


    def get_warped_representations(self, bi_flow, c0, c1, i0=None, i1=None, time_period=0.5):

        # Flow interpolation
        flow_0t = bi_flow[:, :2] * time_period
        flow_1t = bi_flow[:, 2:4] * (1 - time_period)

        # Softsplat Forward Warping
        warped_c0 = softsplat.FunctionSoftsplat(
                tenInput=c0, tenFlow=flow_0t,
                tenMetric=None, strType='average')
        warped_c1 = softsplat.FunctionSoftsplat(
                tenInput=c1, tenFlow=flow_1t,
                tenMetric=None, strType='average')
        
        if (i0 is None) and (i1 is None):
            return warped_c0, warped_c1
        else:
            warped_img0 = softsplat.FunctionSoftsplat(
                    tenInput=i0, tenFlow=flow_0t,
                    tenMetric=None, strType='average')
            warped_img1 = softsplat.FunctionSoftsplat(
                    tenInput=i1, tenFlow=flow_1t,
                    tenMetric=None, strType='average')
            flow_0t_1t = torch.cat((flow_0t, flow_1t), 1)
            return warped_img0, warped_img1, warped_c0, warped_c1, flow_0t_1t


    def forward(self, last_i, i0, i1, c0_pyr, c1_pyr, bi_flow_pyr, time_period=0.5):
        
        # Softsplat Forward Warping & Encoder Network
        warped_img0, warped_img1, warped_c0, warped_c1, flow_0t_1t = \
                self.get_warped_representations(
                        bi_flow_pyr[0], c0_pyr[0], c1_pyr[0], i0, i1,
                        time_period=time_period)
        input_feat = torch.cat(
                (last_i, warped_img0, warped_img1, i0, i1, flow_0t_1t), 1)
        s0 = self.encoder_conv(input_feat)
        s1 = self.encoder_down1(torch.cat((s0, warped_c0, warped_c1), 1))
        
        warped_c0, warped_c1 = self.get_warped_representations(
                        bi_flow_pyr[1], c0_pyr[1], c1_pyr[1],
                        time_period=time_period)
        s2 = self.encoder_down2(torch.cat((s1, warped_c0, warped_c1), 1))
        
        warped_c0, warped_c1 = self.get_warped_representations(
                        bi_flow_pyr[2], c0_pyr[2], c1_pyr[2],
                        time_period=time_period)

        # Decoder Network
        x = self.decoder_up1(torch.cat((s2, warped_c0, warped_c1), 1))
        x = self.decoder_up2(torch.cat((x, s1), 1))
        x = self.decoder_conv(torch.cat((x, s0), 1))

        # prediction
        refine = self.pred(x)
        refine_res = torch.sigmoid(refine[:, :3]) * 2 - 1
        refine_mask0 = torch.sigmoid(refine[:, 3:4])
        refine_mask1 = torch.sigmoid(refine[:, 4:5])

        merged_img = (warped_img0 * refine_mask0 * (1 - time_period) + \
                warped_img1 * refine_mask1 * time_period)
        merged_img = merged_img / (refine_mask0 * (1 - time_period) + \
                refine_mask1 * time_period)

        interp_img = merged_img + refine_res
        interp_img = torch.clamp(interp_img, 0, 1)

        extra_dict = {}
        extra_dict["refine_res"] = refine_res
        extra_dict["warped_img0"] = warped_img0
        extra_dict["warped_img1"] = warped_img1
        extra_dict["merged_img"] = merged_img

        return interp_img, extra_dict



#**************************************************************************************************#
# => Unified model
#**************************************************************************************************#
class Model(nn.Module):
    def __init__(self, pyr_level=3, nr_lvl_skipped=0):
        super(Model, self).__init__()
        self.pyr_level = pyr_level
        self.nr_lvl_skipped = nr_lvl_skipped
        self.feat_pyramid = FeatPyramid()
        self.motion_estimator = raft.raft()
        self.synthesis_network = SynthesisNetwork()

    def forward_one_lvl(self,
            img0, img1, flow_this_lvl, last_interp=None,
            time_period=0.5, skip_me=False):

        # context feature extraction
        feat0_pyr = self.feat_pyramid(img0)
        feat1_pyr = self.feat_pyramid(img1)

        # bi-directional flow estimation
        flow = flow_this_lvl

        # frame synthesis
        ## optical flow is estimated at 1/4 resolution
        ori_resolution_flow = F.interpolate(
                input=flow, scale_factor=4.0,
                mode="bilinear", align_corners=False)

        ## consturct 3-level flow pyramid for synthesis network
        bi_flow_pyr = []
        tmp_flow = ori_resolution_flow
        bi_flow_pyr.append(tmp_flow)
        for i in range(2):
            tmp_flow = F.interpolate(
                    input=tmp_flow, scale_factor=0.5,
                    mode="bilinear", align_corners=False) * 0.5
            bi_flow_pyr.append(tmp_flow)

        ## merge warped frames as initial interpolation for frame synthesis
        if last_interp is None:
            flow_0t = ori_resolution_flow[:, :2] * time_period
            flow_1t = ori_resolution_flow[:, 2:4] * (1 - time_period)
            warped_img0 = softsplat.FunctionSoftsplat(
                    tenInput=img0, tenFlow=flow_0t,
                    tenMetric=None, strType='average')
            warped_img1 = softsplat.FunctionSoftsplat(
                    tenInput=img1, tenFlow=flow_1t,
                    tenMetric=None, strType='average')
            last_interp = warped_img0 * (1 - time_period) \
                    +  warped_img1 * time_period

        ## do synthesis
        interp_img, extra_dict = self.synthesis_network(
                last_interp, img0, img1, feat0_pyr, feat1_pyr, bi_flow_pyr,
                time_period=time_period)
        return interp_img, extra_dict

    def forward(self, img0, img1, time_period,
            pyr_level=None, nr_lvl_skipped=None):
        
        if pyr_level is None: pyr_level = self.pyr_level
        if nr_lvl_skipped is None: nr_lvl_skipped = self.nr_lvl_skipped
        N, _, H, W = img0.shape

        bi_flows = []
        interp_imgs = []
        skipped_levels = [] if nr_lvl_skipped == 0 else\
                list(range(pyr_level))[::-1][-nr_lvl_skipped:]
        
        # Original Flow Estimation: RAFT 추론은 여기서 끝남.
        orig_flow  = self.motion_estimator.pred(img0, img1) # 1/4

        # The original input resolution corresponds to level 0.
        for level in list(range(pyr_level))[::-1]:  # [0, 1, 2]
            if level != 0:
                scale_factor = 1 / 2 ** level       # [1, 1/2, 1/4]
                img0_this_lvl = F.interpolate(
                        input=img0, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False)
                img1_this_lvl = F.interpolate(
                        input=img1, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False)
                flow_this_lvl = F.interpolate(
                    input=orig_flow, scale_factor=scale_factor,
                        mode="bilinear", align_corners=False)
            else:
                img0_this_lvl = img0
                img1_this_lvl = img1
                flow_this_lvl = orig_flow

            # skip motion estimation, directly use up-sampled optical flow
            skip_me = False

            # the lowest-resolution pyramid level
            if level == pyr_level - 1:
                last_flow = F.interpolate(
                    input=flow_this_lvl, scale_factor=1/(2**(level)),
                        mode="bilinear", align_corners=False)
                last_interp = None
            
            # skip some levels for both motion estimation and frame synthesis
            elif level in skipped_levels[:-1]:
                    continue

            # last level (original input resolution), only skip motion estimation
            elif (level == 0) and len(skipped_levels) > 0:
                if len(skipped_levels) == pyr_level:
                    last_flow = flow_this_lvl
                    last_interp = None
                else:
                    resize_factor = 2 ** len(skipped_levels)
                    last_flow = F.interpolate(
                            input=flow_this_lvl, scale_factor=resize_factor,
                            mode="bilinear", align_corners=False) * resize_factor
                    last_interp = F.interpolate(
                            input=interp_img, scale_factor=resize_factor,
                            mode="bilinear", align_corners=False)
                skip_me = True
            
            # last level (original input resolution), motion estimation + frame
            # synthesis
            else:
                last_flow = F.interpolate(input=flow_this_lvl, scale_factor=2.0,
                        mode="bilinear", align_corners=False) * 2
                last_interp = F.interpolate(
                        input=interp_img, scale_factor=2.0,
                        mode="bilinear", align_corners=False)

            # 피라미드 레벨 하나 당 수행하는 Forward 부분.
            interp_img, extra_dict = self.forward_one_lvl(
                    img0_this_lvl, img1_this_lvl,
                    flow_this_lvl, last_interp,
                    time_period, skip_me=skip_me)

            bi_flows.append(
                    F.interpolate(input=flow_this_lvl, scale_factor=4.0,
                        mode="bilinear", align_corners=False))
            interp_imgs.append(interp_img)

        # directly up-sample estimated flow to full resolution with bi-linear
        # interpolation
        bi_flow = F.interpolate(
                input=flow_this_lvl, scale_factor=4.0,
                mode="bilinear", align_corners=False)

        return interp_img, bi_flow, extra_dict['warped_img0'], extra_dict['warped_img1'], \
                {"interp_imgs": interp_imgs, "bi_flows": bi_flows}



if __name__ == "__main__":
    pass
