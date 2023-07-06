"""
RAFT Based Optical Flow Interpolation Warping Method 구현
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class Main_dataloader(Dataset):
    """
    data_root
        >>  00001
            >>  0001
                >>  frame_sequence: 0001.jpg ~ 000N.jpg
    """
    def __init__(self, data_root, mode, stride):

        super().__init__()
        self.mode = mode
        self.datasets = []
        self.data_root = data_root
        self.crop_size = (512,960)
        

        for class_id in os.listdir(self.data_root): # class_id: [00001, ...]

            for label_id in os.listdir(os.path.join(self.data_root, class_id)):     # label_id : [0001, ...]    
                ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root , class_id, label_id)))
                ctg_imgs_ = [os.path.join(self.data_root , class_id, label_id , img_id) for img_id in ctg_imgs_]

                if stride == 1:
                    for start_idx in range(0, len(ctg_imgs_)-2, 3):
                        batch_set = ctg_imgs_[start_idx : start_idx + 3]
                        self.datasets.append(batch_set)
                
                elif stride == 2:
                    for start_idx in range(0, len(ctg_imgs_)-4, 2):
                        batch_set = ctg_imgs_[start_idx : start_idx + 5 : 2]
                        self.datasets.append(batch_set)
                else:
                    pass
        
        self.transforms = transforms.Compose([
            # transforms.CenterCrop(self.crop_size),
            transforms.ToTensor()
            # transforms.Resize(size=self.crop_size),
            # transforms.RandomHorizontalFlip(p=0.5)
            ])

        print(f'{self.mode} RAFT + MiDaS Guide VFI DataLoader: {len(self.datasets)} batch,  input 2 frames output 1 frames')

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        batch_path = self.datasets[idx]
        # print(batch_path) 데이터로더 테스트용
        batch = [Image.open(img) for img in batch_path]
        batch = [self.transforms(img) for img in batch]
        batch = [transforms.functional.resize(img, size=self.crop_size, antialias=False) for img in batch]

        return [batch[0], batch[2]], batch[1], [batch_path[0], batch_path[2]], batch_path[1]

def get_loader(data_root, batch_size, mode, num_workers, stride, shuffle=True):
    dataset = Main_dataloader(data_root, mode=mode, stride=stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    input_img, gt, _, _ = next(iter(dataloader))
    print(f'input: {len(input_img)} x {input_img[0].shape},  gt: {gt.shape},  {input_img[0].dtype},  {input_img[0].mean():.3f}')
    return dataloader

if __name__ == "__main__":
    data_path = "./data/"
    get_loader(data_path, batch_size=8, mode='Test', num_workers=8, shuffle=True)

        


                


