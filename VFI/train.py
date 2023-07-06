"""
RAFT + MiDaS Guide Video Frame Interpolation Train.py
"""

# Library Import ====================================#
import os
import time
import datetime
import shutil
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings(action='ignore')

from dataloader.main_dataloader import get_loader
import config, raft_warping, raft_warping_softsplat_softmax
from API import utils, UNet, metric

print('\n>>>>>> RAFT + MiDaS Guide Video Frame Interpolation <<<<<')
args, unparsed = config.get_args("raft")
cwd = os.getcwd()

torch.cuda.empty_cache()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

writer = SummaryWriter(args.tensorboard_root)

print('\n>>>>>>>>>>>>>>>>>>>>> Initialize <<<<<<<<<<<<<<<<<<<<<<<<')

# DataLoader
train_loader = get_loader(args.data_root + 'train/', args.batch_size, mode='train', num_workers=args.num_workers, stride=2, shuffle=True)
valid_loader = get_loader(args.data_root + 'valid/', args.test_batch_size, mode='valid', num_workers=args.num_workers, stride=2, shuffle=False)

# RAFT + MiDaS + UNet Model Load
device = utils.torch_cuda()
if args.softsplat:
    raft_midas = raft_warping_softsplat_softmax.raft(args, device)
else:
    raft_midas = raft_warping.raft(args, device)

refine_net = UNet.UNet(3,3).to(device)
print(f"Refine Net #params", sum([p.numel() for p in refine_net.parameters()]),'\n')

# Optimizer & Loss & Metric & Scheduler
criterion = metric.Loss(args)
optimizer = metric.adamax(refine_net.parameters())
scheduler = metric.MultiStepLR(optimizer, [10000, 20000, 30000, 40000, 50000, 60000], 0.5)

def main(args):
    print('\n>>>>>>>>>>>>>>>>>>> Train & Valid <<<<<<<<<<<<<<<<<<<<<<')
    best_psnr = 0
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train(args, epoch)
        valid_loss, psnr, ssim = valid(args, epoch)
        metric.log_tensorboard(writer, valid_loss, psnr, ssim, 0, optimizer.param_groups[-1]['lr'], epoch, mode='train')

    # Save Checkpoint
        if ((epoch % args.checkpoint_epoch) == args.checkpoint_epoch - 1):
            torch.save({
                'epoch': epoch,
                'state_dict': refine_net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr': optimizer.param_groups[-1]['lr']
            }, os.path.join(args.checkpoint_root, f'checkpoint{epoch}.pth'))

            # Save best checkpoint
            is_best = psnr > best_psnr
            best_psnr = max(psnr, best_psnr)
            if is_best:
                shutil.copyfile(os.path.join(args.checkpoint_root, f'checkpoint{epoch}.pth'), os.path.join(args.checkpoint_root, 'model_best.pth'))

        one_epoch_time = time.time() - start_time
        metric.print_log(epoch, args.epochs, one_epoch_time, psnr, ssim, optimizer.param_groups[-1]['lr'])
    
    print('\n========= Training Complete =========')

def train(args, epoch):
    losses, psnrs, ssims = metric.init_meters(args.loss)
    refine_net.train()
    criterion.train()

    for i, (input_img, gt_img, input_path, gt_path) in enumerate(train_loader):
        
        # input_path, gt_path : [2,4], [4]
        input_img = [img_.to(device) for img_ in input_img]    # [[4,3,512,960], [4,3,512,960]]
        gt_img = gt_img.to(device)                             # [4,3,512,960]
        optimizer.zero_grad()

        syn_i_gt = raft_midas.exe(input_img, gt_img, input_path, gt_path)  # [B,3,512,960]
        out_img = refine_net(syn_i_gt)                   # [B,3,512,960]

        loss, _ = criterion(out_img, gt_img)
        losses['total'].update(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Save Intermediate Figure
        if 20000 <= i < 20003:
            print(f'syn: {syn_i_gt.mean()}, out: {out_img.mean()}, gt: {gt_img.mean()}')
            utils.viz_img(syn_i_gt, out_img, gt_img, args.out_root, gt_path)

        if i % args.log_iter == 0:
            metric.eval_metrics(out_img, gt_img, psnrs, ssims)

            now = datetime.datetime.now()
            print('{} Train Epoch: {} [{}/{}]\tLoss: {:.4f}\tPSNR: {:.4f}  Lr:{:.4f}'.format(
                now.strftime('%Y-%m-%d %H:%M:%S'), epoch, i+1, len(train_loader), losses['total'].avg, psnrs.avg , optimizer.param_groups[0]['lr'], flush=True))
            
            # Reset metrics
            losses, psnrs, ssims = metric.init_meters(args.loss)

def valid(args, epoch):
    print('Validation for epoch = %d' % epoch)
    losses, psnrs, ssims = metric.init_meters(args.loss)
    refine_net.eval()
    criterion.eval()
    
    with torch.no_grad():
        for i, (input_img, gt_img, _, _) in enumerate(valid_loader):
            
            # input_path, gt_path : [2,4], [4]
            input_img = [img_.to(device) for img_ in input_img]    # [[4,3,512,960], [4,3,512,960]]
            gt_img = gt_img.to(device)                             # [4,3,512,960]

            syn_i_gt = raft_midas.exe(input_img, gt_img, _, _)     # [B,3,512,960]
            out_img = refine_net(syn_i_gt)                         # [B,3,512,960]

            # Save loss values
            loss, loss_specific = criterion(out_img, gt_img)
            for k, v in losses.items():
                if k != 'total':
                    v.update(loss_specific[k].item())
            losses['total'].update(loss.item())

            # Evaluate metrics
            metric.eval_metrics(out_img, gt_img, psnrs, ssims)
    
    return losses['total'].avg, psnrs.avg, ssims.avg

if __name__ == "__main__":
    main(args)
    print('\n>>>>>>>>>>>>>>>>>>>>> Complete <<<<<<<<<<<<<<<<<<<<<<<<')    
