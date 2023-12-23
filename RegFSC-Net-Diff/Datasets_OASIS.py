import os, glob
import torch, sys
from torch.utils.data import Dataset
from Data_utils import pkload
import matplotlib.pyplot as plt
import random
import numpy as np

class OASISBrainDataset3(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)

        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset2(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i, ...] = img == i
        return out

    def __getitem__(self, index):
        # 计算正向和反向索引
        forward_index = index
        backward_index = len(self.paths) * 2 - 1 - index

        if forward_index < len(self.paths):
            # 正向配对：x为当前图像，y为下一个图像
            x_file = self.paths[forward_index]
            y_file = self.paths[(forward_index + 1) % len(self.paths)]
        else:
            # 反向配对：x为当前图像，y为前一个图像
            backward_index = forward_index - len(self.paths)
            x_file = self.paths[backward_index]
            y_file = self.paths[(backward_index - 1) % len(self.paths)]

        x, x_seg = pkload(x_file)
        y, y_seg = pkload(y_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize, channelsHeight, Width, Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize, channelsHeight, Width, Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)

        return x, y, x_seg, y_seg

    def __len__(self):
        # 总长度需要乘以2，因为每对图像都有正向和反向的配对
        return len(self.paths) * 2
