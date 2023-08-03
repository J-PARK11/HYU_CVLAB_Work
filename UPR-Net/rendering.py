import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import shutil
import cv2
import torch
import argparse
import numpy as np
import math
from importlib import import_module

from torch.nn import functional as F
import torchvision.transforms.functional as VF
from core.utils import flow_viz
from core.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_exp_env():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.demo = True

def interp_imgs(ppl, ori_img0, ori_img1):
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)

    width = img0.shape[-1]
    PYR_LEVEL = math.ceil(math.log2(width/448) + 3)
    
    n, c, h, w = img0.shape
    divisor = 2 ** (PYR_LEVEL-1+2)
    
    if (h % divisor != 0) or (w % divisor != 0):
        ph = ((h - 1) // divisor + 1) * divisor
        pw = ((w - 1) // divisor + 1) * divisor
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding, "constant", 0.5)
        img1 = F.pad(img1, padding, "constant", 0.5)
    
    interp_img, bi_flow, warped_img0, warped_img1, extra_dict = ppl.inference(img0, img1,
            time_period=TIME_PERIOID,
            pyr_level=PYR_LEVEL)
    
    interp_img = interp_img[:, :, :h, :w]
    interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)

    return img0, interp_img, img1

def rendering(ppl, data_root, save_dir):
    data_path = sorted([os.path.join(data_root, path) for path in os.listdir(data_root)])
    print(f'Num of Input Frames: {len(data_path)}')
    
    for idx in range(len(data_path)-1):
        img0, img1 = cv2.imread(data_path[idx]), cv2.imread(data_path[idx+1])
        img0, itp_img, img1 = interp_imgs(ppl, img0, img1)
        
        # Save
        save_idx = 2*idx - 1
        if save_idx == 1:
            cv2.imwrite(os.path.join(SAVE_DIR, f'{save_idx:4d}.png'), img0)
        cv2.imwrite(os.path.join(SAVE_DIR, '{(save_idx+1):4d}.png'), itp_img)
        cv2.imwrite(os.path.join(SAVE_DIR, '{(save_idx+2):4d}.png'), img1)
    
    print(f'Num of Output Frames: {len(data_path)*2-1}')
    return None


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--data_root", type=str, required=True,
            help="file path of the input frame")
    parser.add_argument("--time_period", type=float, default=0.5,
            help="time period for interpolated frame")
    parser.add_argument("--save_dir", type=str,
            default="./demo/output",
            help="dir to save interpolated frame")

    # load version of UPR-Net
    parser.add_argument('--model_size', type=str, default="base")
    args = parser.parse_args()

    #**********************************************************#
    # => parse args and init the training environment
    # global variable
    print('\n>>>>>>>>>>>>>>> Rendering.py <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<')
    TIME_PERIOID = args.time_period
    SAVE_DIR = args.save_dir

    # init env
    init_exp_env()
    print(f"Config: {args}")

    #**********************************************************#
    # => init the pipeline and interpolate images
    if args.model_size == 'base':
        model_file = "./checkpoints/upr-base.pkl"
    elif args.model_size == 'large':
        model_file = "./checkpoints/upr-large.pkl"
    elif args.model_size == 'Large':
        model_file = "./checkpoints/upr-llarge.pkl"
    elif args.model_size == 'att':
        model_file = "./checkpoints/upr-att.pkl"
    elif args.model_size == 'raft':
        model_file = "./checkpoints/upr-raft.pkl"
    elif args.model_size == 'depth':
        model_file = "./checkpoints/upr-base.pkl"
    elif args.model_size == 'softmax':
        model_file = "./checkpoints/upr-softmax.pkl"        
    elif args.model_size == 'total':
        model_file = "./checkpoints/upr-total.pkl"        
    elif args.model_size == 'total_depth':
        model_file = "./checkpoints/upr-total-depth.pkl"        
    else:
        ValueError("No mactched Model Size!")

    model_cfg_dict = dict(
            load_pretrain = True,
            model_size = args.model_size,
            model_file = model_file
            )

    ppl = Pipeline(model_cfg_dict)
    ppl.eval()
    print('\n>>>>>>>>>>>>>>>> Rendering <<<<<<<<<<<<<<<<')
    rendering(ppl, args.data_root, args.save_dir)
    print('\n>>>>>>>>>>>>>>>>> Complete <<<<<<<<<<<<<<<<<')
