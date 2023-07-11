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


def interp_imgs(ppl, ori_img0, ori_img1, model_size):
    img0 = (torch.tensor(ori_img0.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    img1 = (torch.tensor(ori_img1.transpose(2, 0, 1)).to(DEVICE) / 255.).unsqueeze(0)
    
    # Only RAFT
    if model_size == 'raft':
        if (img0.shape[-1] > 1000) or (img0.shape[-1] % 8 != 0) or (img0.shape[-2] % 8 != 0):
            crop_size = [512,896]
            img0 = VF.resize(img0, size=crop_size, antialias=False)
            img1 = VF.resize(img1, size=crop_size, antialias=False)
    
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
    
    print(f"\nInput images: {ori_img0.shape} -> {img0.shape}")
    print("Initialization is OK! Begin to interp images")
    
    print('\n>>>>>>>>>>>>>>> Model Inference <<<<<<<<<<<<<<<')
    interp_img, bi_flow, warped_img0, warped_img1 = ppl.inference(img0, img1,
            time_period=TIME_PERIOID,
            pyr_level=PYR_LEVEL)

    print(f'PYR_LEVEL: {PYR_LEVEL},  time_period: {TIME_PERIOID}')
    print(f"Itp Image: {interp_img.shape},  bi_flow: {bi_flow.shape}")
    print(f"Warped img0: {warped_img0.shape},  Warped_img1: {warped_img1.shape}")
    
    interp_img = interp_img[:, :, :h, :w]
    bi_flow = bi_flow[:, :, :h, :w]
    warped_img0 = warped_img0[:, :, :h, :w]
    warped_img1 = warped_img1[:, :, :h, :w]
    
    overlay_input = (ori_img0 * 0.5 + ori_img1 * 0.5).astype("uint8")
    interp_img = (interp_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    bi_flow = bi_flow[0].cpu().numpy().transpose(1, 2, 0)
    warped_img0 = (warped_img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
    warped_img1 = (warped_img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)

    flow01 = bi_flow[:, :, :2]
    flow10 = bi_flow[:, :, 2:]
    flow01 = flow_viz.flow_to_image(flow01, convert_to_bgr=True)
    flow10 = flow_viz.flow_to_image(flow10, convert_to_bgr=True)
    bi_flow = np.concatenate([flow01, flow10], axis=1)

    cv2.imwrite(os.path.join(SAVE_DIR, '0-img0.png'), ori_img0)
    cv2.imwrite(os.path.join(SAVE_DIR, '1-img1.png'), ori_img1)
    cv2.imwrite(os.path.join(SAVE_DIR, '2-overlay-input.png'), overlay_input)
    cv2.imwrite(os.path.join(SAVE_DIR, '3-warped-img0.png'), warped_img0)
    cv2.imwrite(os.path.join(SAVE_DIR, '4-warped-img1.png'), warped_img1)
    cv2.imwrite(os.path.join(SAVE_DIR, '5-interp-img.png'), interp_img)
    cv2.imwrite(os.path.join(SAVE_DIR, '6-bi-flow.png'), bi_flow)
    print("Interpolation is completed! Results in %s" % (SAVE_DIR))
    print('\n>>>>>>>>>>>>>>>>>> Complete <<<<<<<<<<<<<<<<<<')

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
            description="interpolate for given pair of images")
    parser.add_argument("--frame0", type=str, required=True,
            help="file path of the first input frame")
    parser.add_argument("--frame1", type=str, required=True,
            help="file path of the second input frame")
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
    print('\n>>>>>>>>>>>>>>> UPR-Net Demo.py <<<<<<<<<<<<<<<')
    print('\n>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<')
    FRAME0 = args.frame0
    FRAME1 = args.frame1
    TIME_PERIOID = args.time_period
    SAVE_DIR = args.save_dir

    # init env
    init_exp_env()
    print(f"Config: {args}")

    #**********************************************************#
    # => read input frames and calculate the number of pyramid levels
    ori_img0 = cv2.imread(FRAME0)
    ori_img1 = cv2.imread(FRAME1)
    if ori_img0.shape != ori_img1.shape:
        ValueError("Please ensure that the input frames have the same size!")

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
    else:
        ValueError("No mactched Model Size!")

    model_cfg_dict = dict(
            load_pretrain = True,
            model_size = args.model_size,
            model_file = model_file
            )

    ppl = Pipeline(model_cfg_dict)
    ppl.eval()
    interp_imgs(ppl, ori_img0, ori_img1, args.model_size)
