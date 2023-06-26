"""
RAFT Based Optical Flow Interpolation Warping Method 구현
"""

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RAFT_dataloader(Dataset):
    """
    data_root
        >>  ambush1
        >>  ambush5
        >>  glider
        >>  market1
        >>  shaman1
            >>  frame_sequence: 0001.jpg ~ 0002.jpg
    """
    def __init__(self, data_root, mode):

        super().__init__()
        self.mode = mode
        self.datasets = []
        self.data_root = data_root
        self.crop_size = (512,960)
        
        for frame_folder in sorted(os.listdir(self.data_root)):     # [glider, ambush1]
            frame_path = sorted(os.listdir(os.path.join(self.data_root, frame_folder))) # [0001.jpg ~ 000N.jpg]
            frame_path = [os.path.join(self.data_root, frame_folder, img_path) for img_path in frame_path] # Whole Path
            
            for start_idx in range(0, len(frame_path)-2, 3):
                batch_set = frame_path[start_idx : start_idx + 3]
                self.datasets.append(batch_set)
        
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
        batch = [Image.open(img) for img in batch_path]
        batch = [self.transforms(img) for img in batch]
        batch = [transforms.functional.resize(img, size=self.crop_size, antialias=False) for img in batch]

        return [batch[0], batch[2]], batch[1], [batch_path[0], batch_path[2]], batch_path[1]

def get_loader(data_root, batch_size, mode, num_workers, shuffle=True):
    dataset = RAFT_dataloader(data_root, mode=mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    input_img, gt, _, _ = next(iter(dataloader))
    print(f'input: {len(input_img)} x {input_img[0].shape},  gt: {gt.shape},  {input_img[0].dtype},  {input_img[0].mean():.3f}')
    return dataloader

if __name__ == "__main__":
    data_path = "./data/"
    get_loader(data_path, batch_size=8, mode='Test', num_workers=8, shuffle=True)

        


                


