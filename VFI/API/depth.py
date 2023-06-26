import os
import time
import datetime
import argparse
from math import log10

import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

def Midas_depth(model_type='DPT_Large'):
    #Midas Model Load from different type
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Freeze Midas Parameter.
    for name, param in midas.named_parameters():
        param.requries_grad = False
    
    # Transform Function.
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    # Return Model & Transformation Function.
    return midas, transform

def midas_pred(midas, img):
    with torch.no_grad():
        prediction = midas(img)

        # prediction = torch.nn.functional.interpolate(
        #     prediction.unsqueeze(1),
        #     size=raw_img.shape[2:],
        #     mode="bicubic",
        #     align_corners=False,
        # ).squeeze()

    return prediction