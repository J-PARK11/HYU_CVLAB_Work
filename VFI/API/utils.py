import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
import torchvision.transforms.functional as F

import moviepy
import moviepy.editor
import moviepy.video.io.ffmpeg_writer

frame_in = 'data/boxing/'
video_in = 'data/videos/car-turn.mp4'
video_out = 'data/output/boxing.mp4'
frame_out = 'data/output/car-turn/'

# Load image from folder
def load_image(img_path):
    dataset = []
    for idx, image in enumerate(img_path):
        img = Image.open(image)
        img = np.array(img).astype(float) / 255.0
        dataset.append(img)
    dataset = np.array(dataset)
    return dataset

def save_flow(flow, path):
    flow_img = flow_to_image(flow)
    flow_img = F.to_pil_image(flow_img)
    plt.imshow(np.asarray(flow_img), cmap='hsv')
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    # cv2.imwrite(path, cv2.cvtColor(np.asarray(flow_img), cv2.COLOR_BGR2HSV))

def save_img(warped, path):
    plt.imshow(np.asarray(warped).clip(0,1))
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    # cv2.imwrite(path, cv2.cvtColor(np.asarray(warped)*255.0, cv2.COLOR_BGR2RGB))

def save_depth(depth, path):
    plt.imshow(np.asarray(depth))
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    # cv2.imwrite(path, cv2.cvtColor(np.asarray(warped)*255.0, cv2.COLOR_BGR2RGB))

# Visualize Interpolation
def vis_interpolation(input_path, output_path):
    fig, axes = plt.subplots(1,3,figsize=(10,4))
    input1 = Image.open(input_path[0])
    input2 = Image.open(input_path[1])
    output = Image.open(output_path)
    for idx, img in enumerate([input1, output, input2]):
        label = ['t-1', 't', 't+1']
        axes[idx].imshow(img)
        axes[idx].set_title(f'{label[idx]}')
        axes[idx].axis(('off'))

# Generate Frame to Video
def frame_to_video(in_path, out_path, fps):
    frame_list = [in_path + file for file in sorted(os.listdir(in_path))]
    print(f'Name: {frame_list[0]} len: {len(frame_list)}')
    clip = moviepy.editor.ImageSequenceClip(frame_list, fps)
    clip.write_videofile(out_path, fps)
    print('end')
    clip.close()

# Generate Video to Frame
def video_to_frame(in_path, out_path, fps):
    clip = moviepy.editor.VideoFileClip(in_path)
    print(f'Video = duration: {clip.duration} fps: {clip.fps} ')
    iter = int(clip.duration * fps)
    print(f'frames = len: {iter} fps: {fps}')
    for t in range(iter):
        clip.save_frame(f'{out_path}/frame{t+1}.png', t = 1/fps*t)
    print('end')
    clip.close()

# Pytorch CUDA Connection
def torch_cuda():
    av = torch.cuda.is_available()
    cr = torch.cuda.current_device()
    lg = torch.cuda.device_count()
    name = torch.cuda.get_device_name(cr)
    print(f'CUDA available {av},  Usuable devices: {lg},   Current device: {cr},  name: {name}\n')
    device = torch.device('cuda')
    return device

def viz_img(syn, out, gt, out_root, gt_path, save=True):
        
        for i, path in enumerate(gt_path):

            # Save format    
            save_syn = syn.cpu().detach()[i].permute(1,2,0).clip(0,1)
            save_out = out.cpu().detach()[i].permute(1,2,0).clip(0,1)
            save_gt = gt.cpu().detach()[i].permute(1,2,0).clip(0,1)

            # Save
            idx = path.split('/')[-2]
            name = path.split('/')[-1]

            save_img(save_syn, out_root + idx + '_syn_' + name)
            save_img(save_out, out_root + idx + '_out_' + name)
            save_img(save_gt, out_root + idx +'_gt_' + name)

def viz_test_img(syn, gt, out_root, gt_path, save=True):
        
        for i, path in enumerate(gt_path):

            # Save format    
            save_syn = syn.cpu().detach()[i].permute(1,2,0).clip(0,1)
            save_gt = gt.cpu().detach()[i].permute(1,2,0).clip(0,1)

            # Save
            idx = path.split('/')[-2]
            name = path.split('/')[-1]

            save_img(save_syn, out_root + idx + '_syn_' + name)
            save_img(save_gt, out_root + idx +'_gt_' + name)

# Flow 불러오는 함수 --------------------------------------------------- #
def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()

    assert(np.frombuffer(buffer=strFlow, dtype=np.float32, count=1, offset=0))

    intWidth = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=4)[0]
    intHeight = np.frombuffer(buffer=strFlow, dtype=np.int32, count=1, offset=8)[0]
    
    intflo = np.frombuffer(buffer=strFlow, dtype=np.float32, count=intHeight * intWidth * 2, offset=12).reshape(intHeight, intWidth, 2)
    print(f'read_flo shape :{intflo.shape}  read_flo type :{intflo.dtype}')
    return intflo
# ------------------------------------------------------------------- #

# -------------------- test ----------------------- #  
# frame_to_video(frame_in, video_out, 40)
# video_to_frame(video_in, frame_out, 40)