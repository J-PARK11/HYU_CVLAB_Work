import os
import cv2
import time
import torch
import ctypes
import argparse
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt

from class_dict import *
from ultralytics import YOLO

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', default='./ultralytics/sample4/', type=str, help='Image data root')
    parser.add_argument('--output_root', default='./output/temp/', type=str, help='Output root')
    parser.add_argument('--real_time', default= False, type=bool, help='Real Time arg')
    return parser

class yolo:
    def __init__(self, args):
        super(yolo, self).__init__()
        self.args = args
        self.config = self.args.__dict__

        self.data_root = args.data_root
        self.real_time = args.real_time
        self.output_root = args.output_root
        self.data_path = self.load_data_path()
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('yolov8n-seg.pt')
        print('Load Yolo Model Success')
        return model
    
    def load_data_path(self):
        data_path = [os.path.join(self.data_root, img_path) for img_path in os.listdir(self.data_root)]
        print(f'Load data_path: {len(data_path)} images Success')
        return data_path

    def get_binary_label(self):
        bin_label = []
        for img in self.pred:
            cls = img.boxes.cls
            batch = []
            class_label = []
            for key in cls:
                ele = dynamic_obj(key)
                batch.append(ele)
                ele_class = name_dict(int(key.cpu().detach()))
                class_label.append(ele_class)
            bin_label.append(batch)
            print(f'class_label: {class_label}')
            print(f'orig_label: {cls}')
            print(f'bin_label: {batch}\n')
        return bin_label
    
    def get_dynamic_segmentation(self):
        dynamic_output, dynamic_pred = [], []
        idx = 0
        for img, bin_label in zip(self.pred, self.bin_label):
            pallete = torch.tensor(np.zeros(img.masks.data[0].shape))
            for ele_mask, ele_label in zip(img.masks.data, bin_label):
                if ele_label == 1:
                    pallete += ele_mask.cpu().detach()
                else:
                    pass
            
            # resizing
            output = pallete.clip(0,1).numpy()
            pallete = cv2.resize(output, dsize=(img.orig_shape[1], img.orig_shape[0]), interpolation = cv2.INTER_NEAREST)
            plt.imshow(pallete)
            plt.savefig(f'{self.output_root}{idx}_mask.png', bbox_inches='tight')
            idx += 1
            dynamic_output.append(output)
            dynamic_pred.append(pallete)
        
        print(f'dynamic_pred: {len(dynamic_pred)} images   {dynamic_pred[0].shape}')
        print(f'dynamic_output: {len(dynamic_output)} list[numpy]   {dynamic_output[0].shape}, {dynamic_output[0].dtype}, {dynamic_output[0].mean()}')
        return dynamic_pred, dynamic_output

    def comparison_orig_pred(self):
        idx = 0
        for orig, pred in zip(self.pred, self.dynamic_pred):
            fig, ax = plt.subplots(1,2, figsize=(10, 6))
            ax[0].imshow(cv2.cvtColor(orig.orig_img, cv2.COLOR_BGR2RGB))
            ax[1].imshow(pred)
            plt.savefig(f'{self.output_root}{idx}_comparison.png', bbox_inches='tight')
            idx += 1    
    
    def np_to_ctypes_array(self):
        
        """
        Convert a NumPy array to a ctypes array.

        Parameters:
        - arr: NumPy array

        Returns:
        - ctypes array
        """

         # Get the data type and shape of the array
        arr = self.dynamic_output[0]
        arr_dest = arr.copy()
        dtype = arr.dtype
        shape = arr.shape

        # Create a ctypes array with the same shape and data type
        if dtype == np.float32:
            c_type = ctypes.c_float
        elif dtype == np.float64:
            c_type = ctypes.c_double
        elif dtype == np.int32:
            c_type = ctypes.c_int32
        elif dtype == np.int64:
            c_type = ctypes.c_int64
        else:
            raise ValueError("Unsupported data type: {}".format(dtype))
        
        # 포인터 자료형을 정의한다.
        c_pointer = ctypes.POINTER(c_type)

        # 포인터 변수에 원본 데이터의 주소를 받는다.
        c_arr = arr.ctypes.data_as(c_pointer)
        c_arr_dest = arr_dest.ctypes.data_as(c_pointer)
        print(c_arr, type(c_arr))
        print(c_arr_dest, type(c_arr_dest))

        # Load Dynamic Library
        # c_lib = ctypes.CDLL(self.args.output_root + 'dynamic_output_ctypes.so')
        # c_lib.dynamic_output_ctypes(c_arr, c_arr_dest, len(arr))
        return c_arr

    def exe(self):
        
        if self.real_time:
            self.pred = self.model.predict(source='0', show=True)
        else:
            start_time = time.time()
            self.pred = self.model(self.data_path)
            end_time = time.time()
            du = end_time - start_time
            print(f"total duration: {du:.6f}   image per duration: {du/len(self.pred):.6f}")

            print(f'\nSegmentation Results: {len(self.pred)} images')
            print(f'orig_shape: {self.pred[0].orig_shape} --> mask_shape: {self.pred[0].masks.data.shape}\n')
        
        self.bin_label = self.get_binary_label()
        self.dynamic_pred, self.dynamic_output = self.get_dynamic_segmentation()
        self.comparison_orig_pred()

        ctype_result = self.np_to_ctypes_array()


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__
    
    print('======= Initialize =======\n')
    exe = yolo(args)
    exe.exe()
    print('\n======== Complete ========')

"""

김범수 100
partition 주짓수 30 / SLAM 50 / 옵치 20

"""
