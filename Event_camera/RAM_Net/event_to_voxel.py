from utils.event_tensor_utils import *
import cv2
import os
import shutil

def convert_from_event_to_voxel(path: str):
    files = os.listdir(path)
    width = 512
    height = 256
    data_path = os.path.join(path, 'data')
    boundary_list = []
    timestamp_list = []
    new_path = os.path.join(path, 'voxels')
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    # t0 = -1
    i = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_name = os.path.join(root, file)
            if file == 'boundary_timestamps.txt':
                shutil.copyfile(file_name, os.path.join(new_path, file))
                f = open(file_name, 'r')
                lines = f.readlines()
                for line in lines:
                    boundary_list.append(line.strip().split(' '))
            elif file == 'timestamps.txt':
                shutil.copyfile(file_name, os.path.join(new_path, file))
                f = open(file_name, 'r')
                lines = f.readlines()
                for line in lines:
                    timestamp_list.append(line.strip().split(' '))
            else:
                events = np.load(file_name)
                x, y, p, t = events['x'], events['y'], events['p'], events['t']
                # if t0 < 0:
                #     t0 = t[0]
                # t = t - t0
                t = t / 1e6
                single_np_event = np.vstack([t, x, y, p]).T
                voxel = events_to_voxel_grid(single_np_event, 5, width, height)
                # np.save(os.path.join(new_path, '05_{:03d}_{:04d}_voxel.npy'.format(name_idx, i)), voxel)
                new_file = file.replace('events', 'voxel').replace('npz', 'npy')
                np.save(os.path.join(new_path, new_file), voxel)
                i += 1
                pass

if __name__ == '__main__':
    file_path = '/home/work/main/jpark/Event_camera/data/Town03/'
    for f in sorted(os.listdir(file_path)):
        idx = f.split('_')[-1]
        dir = os.path.join(file_path, f) + '/events'
        convert_from_event_to_voxel(dir)
        print(dir)