# -*- coding: utf-8 -*-
"""
EventDataset classes
>>>
EventDataet(base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True)
VoxelGridDataset(base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True)
RawEventsDataset(base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True)
FrameDataset(base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True)
---
initial_stamp: 실제 시작하는 타임스탬프 값.
first_stamp: 상대적인 첫 타임스탬프 값.
last_stamp: 상대적인 마지막 타임스탬프 값.
length: 총 타임스탬프 갯수 = Event가 기록된 프레임 갯수.
stamps: 상대적인 타임스탬프 진행 값. 
---
event_tensor will be a [num_bins x H x W] floating point array
>>>
VoxelGridDataset: torch.Size([5, 256, 512])
RawEventsDataset: (5, 256, 512)
FrameDataset: torch.Size([1, 256, 512])
"""

from torch.utils.data import Dataset
from skimage import io
from os.path import join
import numpy as np
from utils.util import first_element_greater_than, last_element_less_than
import random
import torch
import glob


class EventDataset(Dataset):
    """Loads event tensors from a folder, with different event representations."""
    def __init__(self, base_folder, event_folder, start_time=0, stop_time=0, transform=None, normalize=True):
        self.base_folder = base_folder
        self.event_folder = join(self.base_folder, event_folder)
        self.transform = transform

        self.start_time = start_time
        self.stop_time = stop_time

        self.normalize = normalize

        if "mvsec" in base_folder or "javi" in base_folder:
            self.use_mvsec = True
        else:
            self.use_mvsec = False

        self.read_timestamps()
        self.parse_event_folder()
        self.data_id = int(self.base_folder.split('/')[-1][4:6])
        self.seq_id = int(self.base_folder.split('/')[-1].split('_')[-1])

    def read_timestamps(self):
        # Load the timestamps file
        raw_stamps = np.loadtxt(join(self.event_folder, 'timestamps.txt'))

        if raw_stamps.size == 0:
            raise IOError('Dataset is empty')

        if len(raw_stamps.shape) == 1:
            # if timestamps.txt has only one entry, the shape will be (2,) instead of (1, 2). fix that.
            raw_stamps = raw_stamps.reshape((1, 2))

        self.stamps = raw_stamps[:, 1]
        if self.stamps is None:
            raise IOError('Unable to read timestamp file: '.format(join(self.event_folder,
                                                                        'timestamps.txt')))

        # Check that the timestamps are unique and sorted
        assert(np.alltrue(np.diff(self.stamps) > 0)), "timestamps are not unique and monotonically increasing"

        self.initial_stamp = self.stamps[0]
        self.stamps = self.stamps - self.initial_stamp  # offset the timestamps so they start at 0

        # Find the index of the first event tensor whose timestamp >= start_time
        # If there is none, throw an error
        if self.start_time <= 0.0:
            self.first_valid_idx, self.first_stamp = 0, self.stamps[0]
        else:
            self.first_valid_idx, self.first_stamp = first_element_greater_than(self.stamps, self.start_time)
        assert(self.first_stamp is not None)
        # print('First valid index / stamp = {}, {}'.format(self.first_valid_idx, self.first_stamp))

        # Find the index of the last event tensor whose timestamp <= end_time
        # If there is None, throw an error
        if self.stop_time <= 0.0:
            self.last_valid_idx, self.last_stamp = len(self.stamps) - 1, self.stamps[-1]
        else:
            self.last_valid_idx, self.last_stamp = last_element_less_than(self.stamps, self.stop_time)
        assert(self.last_stamp is not None)
        # print('Last valid index / stamp = {}, {}'.format(self.last_valid_idx, self.last_stamp))

        assert(self.first_stamp <= self.last_stamp)

        if self.use_mvsec and not "javi" in self.base_folder:
            self.length = self.last_valid_idx - self.first_valid_idx + 1 - 1
        else:
            self.length = self.last_valid_idx - self.first_valid_idx + 1
        assert(self.length > 0)

    def parse_event_folder(self):
        """Parses the event folder to check its validity and read the parameters of the event representation."""
        raise NotImplementedError

    def __len__(self):
        return self.length

    def get_last_stamp(self):
        """Returns the last event timestamp, in seconds."""
        return self.stamps[self.last_valid_idx]

    def num_channels(self):
        """Returns the number of channels of the event tensor."""
        raise NotImplementedError

    def get_index_at(self, i):
        """Returns the index of the ith event tensor"""
        return self.first_valid_idx + i

    def get_stamp_at(self, i):
        """Returns the timestamp of the ith event tensor"""
        return self.stamps[self.get_index_at(i)]

    def __getitem(self, i):
        """Returns a C x H x W event tensor for the ith element in the dataset."""
        raise NotImplementedError


class VoxelGridDataset(EventDataset):
    """Load an event folder containing event tensors encoded with the VoxelGrid format."""

    def parse_event_folder(self):
        """Check that the passed directory has the following form:

        ├── event_folder
        |   ├── timestamps.txt
        |   ├── event_tensor_0000000000.npy
        |   ├── ...
        |   ├── event_tensor_<N>.npy
        """
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        # event_tensor will be a [num_bins x H x W] floating point array
        # event_tensor = np.load(join(self.event_folder, 'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i)))
        if self.use_mvsec:
            event_tensor = np.load(join(self.event_folder, 'event_tensor_{:010d}.npy'.format(self.first_valid_idx + i)))
        else:
            path_event = glob.glob(self.event_folder + '/*_{:04d}_voxel.npy'.format(self.first_valid_idx + i))
            event_tensor = np.load(path_event[0])[-2:]
        
        if self.normalize:
            # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
            # in the tensor are equal to (0.0, 1.0)
            mask = np.nonzero(event_tensor)
            if mask[0].size > 0:
                mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
                if stddev > 0:
                    event_tensor[mask] = (event_tensor[mask] - mean) / stddev

        self.num_bins = event_tensor.shape[0]

        events = torch.from_numpy(event_tensor)  # [C x H x W]
        if self.transform:
            random.seed(transform_seed)
            events = self.transform(events)

        ## Hand Made =========================== #########################
        """
        negativ_input = np.zeros(events.shape[1:])
        positiv_input = np.zeros(events.shape[1:])
        
        for weight, eve in zip([1.0, 2.0, 3.0, 4.0, 5.0], events):
            negativ = np.where(eve <= -0.6, -weight, 0.0)
            positiv = np.where(eve >= 0.6, weight, 0.0)
            negativ_input[negativ != 0 ] = negativ[negativ != 0 ]
            positiv_input[positiv != 0 ] = positiv[positiv != 0 ]
        
        negativ_input = torch.tensor(negativ_input).unsqueeze(0).type(torch.FloatTensor)
        positiv_input = torch.tensor(positiv_input).unsqueeze(0).type(torch.FloatTensor)
        
        events = torch.cat([events, negativ_input, positiv_input], axis=0)
        """
        return {'events': events}  # [num_bins x H x W] tensor

class RawEventsDataset(EventDataset):
    """Load an event folder containing event tensors encoded with the VoxelGrid format."""

    def parse_event_folder(self):
        """Check that the passed directory has the following form:
        ├── event_folder
        |   ├── timestamps.txt
        |   ├── events_0000000000.npy
        |   ├── ...
        |   ├── events_<N>.npy
        """
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        # events_raw will be a [timestamp, x, y, polarity] floating point array
        # events_raw = np.load(join(self.event_folder, 'events_{:010d}.npy'.format(self.first_valid_idx + i)))
        # path_event = glob.glob(self.event_folder + '/*_{:04d}_events.npy'.format(self.first_valid_idx + i))
        path_event = glob.glob(self.event_folder + '/*_{:04d}_voxel.npy'.format(self.first_valid_idx + i))
        events_raw = np.load(path_event[0])
        return events_raw


class FrameDataset(EventDataset):
    """Load a frame folder containing RGB images."""

    def parse_event_folder(self):
        """Check that the passed directory has the following form:

        ├── frame_folder
        |   ├── timestamps.txt
        |   ├── frame_0000000000.png
        |   ├── ...
        |   ├── frame_<N>.png
        """
        self.frame_folder = self.event_folder #work around
        self.num_bins = None

    def num_channels(self):
        return self.num_bins

    def __getitem__(self, i, transform_seed=None):
        assert(i >= 0)
        assert(i < self.length)

        if transform_seed is None:
            transform_seed = random.randint(0, 2**32)

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.float32)

        # frame_tensor will be a [1 x H x W] floating point array (grayscale images)
        #print("first_valid_idx: ", self.first_valid_idx)
        #print("index for rgb image:", self.first_valid_idx + i, "at index ", i)
        rgb_frame = io.imread(join(self.frame_folder, '{:02d}_{:03d}_{:04d}_image.png'.format(self.data_id, self.seq_id, self.first_valid_idx + i)), as_gray=False).astype(np.float32)
        if rgb_frame.shape[2] > 1:
            frame_tensor = rgb2gray(rgb_frame) #[H x W]
        
        if self.normalize:
            frame_tensor /= 255.0 #normalize
            frame_tensor = np.expand_dims(frame_tensor, axis=0) #expand to [1 x H x W]

        self.num_bins = frame_tensor.shape[0]

        frames = torch.from_numpy(frame_tensor)  # [1 x H x W]
        if self.transform:
            random.seed(transform_seed)
            frames = self.transform(frames)

        return {'frames': frames}  # [1 x H x W] tensor

if __name__ == '__main__':
    event = VoxelGridDataset(base_folder="/home/work/main/jpark/Event_camera/data/Town05_test/sequence_0",
                            event_folder="events/voxels",
                            start_time=0, stop_time=0, transform=None, normalize=True)
    voxel = event.__getitem__(0)['events']
    print(f'VoxelGridDataset: {voxel.shape}')
    
    event = RawEventsDataset(base_folder="/home/work/main/jpark/Event_camera/data/Town05_test/sequence_0",
                            event_folder="events/voxels",
                            start_time=0, stop_time=0, transform=None, normalize=True)
    voxel = event.__getitem__(0)
    print(f'RawEventsDataset: {voxel.shape}')

    rgb = FrameDataset(base_folder="/home/work/main/jpark/Event_camera/data/Town05_test/sequence_0",
                            event_folder="rgb/data",
                            start_time=0, stop_time=0, transform=None, normalize=True)
    frame = rgb.__getitem__(0)['frames']
    print(f'FrameDataset: {frame.shape}')