import os
from PIL import Image
import os.path as osp
import numpy as np
import argparse
import cv2

import torch
import torchvision
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings('ignore')

class raft:

    def __init__(self):
        super(raft, self).__init__()
        self._build_model()

    def _build_model(self):
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Raft Flow Model & Weight load
        self.weight = torchvision.models.optical_flow.Raft_Large_Weights.C_T_V2
        self.model = torchvision.models.optical_flow.raft_large(self.weight).to(self.DEVICE)

        # self.weight = torchvision.models.optical_flow.Raft_Small_Weights.C_T_V2
        # self.model = torchvision.models.optical_flow.raft_small(self.weight).to(DEVICE)

    def pred(self, img0, img1):
        
        self.model.eval()
        with torch.no_grad():
            
            # Flow Estimation
            self.f01 = self.model(img0, img1)[-1]       # [B,2,H,W]
            self.f10 = self.model(img1, img0)[-1]       # [B,2,H,W]

            self.bi_flow = torch.concat([self.f01, self.f10], dim=1)
            # self.bi_flow = torch.nn.functional.interpolate(self.bi_flow, scale_factor=0.25, mode="bilinear", align_corners=False)
  
        return self.bi_flow
    
    def dataload(self, path0, path1):
        img0 = torch.tensor(self.load_image([path0]).astype('float32')).swapaxes(1,3).swapaxes(2,3).to(self.DEVICE)
        img1 = torch.tensor(self.load_image([path1]).astype('float32')).swapaxes(1,3).swapaxes(2,3).to(self.DEVICE)
        
        if (img0.shape[-1] > 1000) or (img0.shape[-1] % 8 != 0) or (img0.shape[-2] % 8 != 0):
            crop_size = [512,896]   # [520, 960]
            img0 = F.resize(img0, size=crop_size, antialias=False)
            img1 = F.resize(img1, size=crop_size, antialias=False)

        return img0, img1
    
    def load_image(self, img_path):
        dataset = []
        for idx, image in enumerate(img_path):
            img = Image.open(image)
            img = np.array(img).astype(float) / 255.0
            dataset.append(img)
        dataset = np.array(dataset)
        return dataset

if __name__ == '__main__':
    
    path0 = '/home/work/main/jpark/UPR-Net/demo/images/glider0.png'
    path1 = '/home/work/main/jpark/UPR-Net/demo/images/glider1.png'
    SAVE_DIR = "/home/work/main/jpark/UPR-Net/demo/output"

    print('\n>>>>>>>>>>>> RAFT Flow based Warping Start <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')
    exe = raft()
    img0, img1 = exe.dataload(path0, path1)
    print('\n>>>>>>>>>>>>>>>>> Main Implementation <<<<<<<<<<<<<<<<<<<<')
    bi_flow = exe.pred(img0, img1)[0].cpu().numpy().transpose(1, 2, 0)
    
    import flow_viz
    flow01 = bi_flow[:, :, :2]
    flow10 = bi_flow[:, :, 2:]
    flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
    flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
    bi_flow = np.concatenate([flow01, flow10], axis=1)

    cv2.imwrite(os.path.join(SAVE_DIR, 'raft_bi-flow.png'), bi_flow)


    print('\n>>>>>>>>>>>>>>>>>>>>>>>> End <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')



