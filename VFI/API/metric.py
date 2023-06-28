"""
Loss & Optimizer & Metric module
"""

import time
import math
import torch
import torch.nn as nn
from pytorch_msssim import ssim as calc_ssim

# Loss Definition ============================#
def l1_loss(): # MAE
    return nn.L1Loss()

def l2_loss(): # MSE
    return nn.MSELoss()

def Huber_loss():   # Huber Loss
    return nn.HuberLoss(delta=.5)

class Loss(nn.modules.loss._Loss):      # Wrapper of Loss Functions: (args.loss: "1*l1+0.5*l2+1*Huber")
    def __init__(self, args):
        super(Loss, self).__init__()

        # Loss Component 
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):   # [1*l1, 0.5*l2, 1*Huber]
            weight, loss_type = loss.split('*')
            if loss_type == 'l2':
                loss_function = l2_loss()
            elif loss_type == 'l1':
                loss_function = l1_loss()
            elif loss_type == 'Huber':
                loss_function = Huber_loss()
            else:
                continue
            
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function
                })
        
        # Total Function
        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})
        
        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        # Feed to cuda 
        device = torch.device('cuda' if args.cuda else 'cpu')
        self.loss_module.to(device)
        self.loss_module = nn.DataParallel(self.loss_module)

    # ****** 함수는 위에서 정의하고 여기서 원하는 예측값과 GT를 통해 Loss를 계산해야한다. ******
    def forward(self, pred, gt):
        total_loss = 0
        losses = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                _loss = l['function'](pred, gt)
            weighted_loss = l['weight'] * _loss
            losses[l['type']] = weighted_loss
            total_loss += weighted_loss

        return total_loss, losses
            
# Optimizer Definition ============================#
# Optimizer
def adam(params, lr=0.002, betas=(0.9, 0.999), eps=1e-08):
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)

def adamax(params, lr=0.002, betas=(0.9, 0.999)):  
    return torch.optim.Adamax(params, lr=lr, betas=betas)

# Scheduler
def OneCycleLR(optim, len_batch, epochs):   # len(dataloader), len(epochs)
    return torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.1, steps_per_epoch=len_batch, epochs=epochs)

def MultiStepLR(optim, milestones=[100, 150], gamma=0.1):
    return torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=gamma)

# Metric Definition ============================#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_losses(loss_str):
    loss_specifics = {}
    loss_list = loss_str.split('+')
    for l in loss_list:
        _, loss_type = l.split('*')
        loss_specifics[loss_type] = AverageMeter()
    loss_specifics['total'] = AverageMeter()
    return loss_specifics

def init_meters(loss_str):
    losses = init_losses(loss_str)
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return losses, psnrs, ssims

def eval_metrics(output, gt, psnrs, ssims):
    # PSNR should be calculated for each image, since sum(log) =/= log(sum).
    for b in range(gt.size(0)):
        psnr = calc_psnr(output[b], gt[b])
        psnrs.update(psnr)

        ssim = calc_ssim(output[b].unsqueeze(0).clamp(0,1), gt[b].unsqueeze(0).clamp(0,1))
        ssims.update(ssim)

def calc_psnr(pred, gt):
    diff = (pred - gt).pow(2).mean() + 1e-8
    return -10 * math.log10(diff)

# Train Logging ============================#
def log_tensorboard(writer, loss, psnr, ssim, lpips, lr, timestep, mode='train'):
    writer.add_scalar(f'Loss/{mode}', loss, timestep)
    writer.add_scalar(f'PSNR/{mode}', psnr, timestep)
    writer.add_scalar(f'SSIM/{mode}', ssim, timestep)
    if mode == 'train':
        writer.add_scalar(f'lr/{mode}', lr, timestep)

def print_log(epoch, num_epochs, one_epoch_time, oup_pnsr, oup_ssim, Lr):
    print('({0:.0f}s) Epoch [{1}/{2}], Val_PSNR:{3:.2f}, Val_SSIM:{4:.4f}'
          .format(one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim))
    # write training log
    with open('./version/train_log.txt', 'a') as f:
        print(
            'Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}, Lr:{6:.4f}'
            .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    one_epoch_time, epoch, num_epochs, oup_pnsr, oup_ssim, Lr), file=f)