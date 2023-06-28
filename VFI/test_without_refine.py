"""
RAFT + MiDaS Guide Video Frame Interpolation without RefineNet Test.py
"""

# Library Import ====================================#
import os
import time
import torch
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

from dataloader.main_dataloader import get_loader
import raft_warping, config
from API import utils, UNet, metric

print('\n>>>>>>>> RAFT + MiDaS Guide without RefineNet <<<<<<<')
args, unparsed = config.get_args("raft")
cwd = os.getcwd()

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')

# DataLoader
test_loader = get_loader(args.data_root + 'test/', args.test_batch_size, mode='test', num_workers=args.num_workers, stride=2, shuffle=False)
# test_loader = get_loader('data/', args.test_batch_size, mode='test', num_workers=args.num_workers, stride=1, shuffle=False)

# RAFT + MiDaS Model Load
device = utils.torch_cuda()
raft_midas = raft_warping.raft(args, device)

# Test loop ========================================#
def test(args):
    time_taken = []
    losses, psnrs, ssims = metric.init_meters(args.loss)

    with torch.no_grad():
        for i, (input_img, gt_img, input_path, gt_path) in enumerate(tqdm(test_loader)):
            
            # input_path, gt_path : [2,4], [4]
            input_img = [img_.to(device) for img_ in input_img]    # [[4,3,512,960], [4,3,512,960]]
            gt_img = gt_img.to(device)                             # [4,3,512,960]

            start_time = time.time()
            syn_i_gt = raft_midas.exe(input_img, gt_img, input_path, gt_path)     # [B,3,512,960]
            time_taken.append(time.time() - start_time)

            metric.eval_metrics(syn_i_gt, gt_img, psnrs, ssims)

            # Save Figure
            # if (args.test_batch_size * (i+1) <= 50):
            #     utils.viz_test_img(syn_i_gt, gt_img, args.out_root, gt_path)

    print("PSNR: %f, SSIM: %fn" %
          (psnrs.avg, ssims.avg))
    print("Time , " , sum(time_taken)/len(time_taken))

    return psnrs.avg     

# Main Paragraph ====================================#        
def main(args):
    test(args)

# Execute ===========================================#        
if __name__ == "__main__":
    main(args)
    print('\n>>>>>>>>>>>>>>>>>>>>> Complete <<<<<<<<<<<<<<<<<<<<<<<<')