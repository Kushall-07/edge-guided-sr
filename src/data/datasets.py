import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

def read_tif(path):
    with rasterio.open(path) as ds:
        arr = ds.read().astype(np.float32)
    return arr

class PairedThermalOpticalDataset(Dataset):
    def __init__(self, thermal_dir, optical_dir, tile_size=96, stride=64, scale=3, split='train', split_ratio=0.9):
        self.tile_size = tile_size
        self.stride = stride
        self.scale = scale
        tifs = sorted(glob.glob(os.path.join(thermal_dir, '*.tif')))
        pairs = []
        for t in tifs:
            name = os.path.splitext(os.path.basename(t))[0]
            o = os.path.join(optical_dir, name + '.tif')
            if os.path.isfile(o):
                pairs.append((t,o))
        n = len(pairs)
        n_train = int(n*split_ratio)
        self.pairs = pairs[:n_train] if split=='train' else pairs[n_train:]
        self.indices = []
        for t_path, o_path in self.pairs:
            with rasterio.open(o_path) as ds:
                H, W = ds.height, ds.width
            for y in range(0, H - tile_size + 1, stride):
                for x in range(0, W - tile_size + 1, stride):
                    self.indices.append((t_path, o_path, y, x))
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        t_path, o_path, y, x = self.indices[i]
        with rasterio.open(o_path) as dso:
            o = dso.read().astype(np.float32)
        with rasterio.open(t_path) as dst:
            t = dst.read().astype(np.float32)
        o_tile = o[:, y:y+self.tile_size, x:x+self.tile_size]
        yl, xl = y//self.scale, x//self.scale
        hl, wl = self.tile_size//self.scale, self.tile_size//self.scale
        t_tile = t[:, yl:yl+hl, xl:xl+wl]
        return {'t_lr': torch.from_numpy(t_tile).float(), 'o_hr': torch.from_numpy(o_tile).float()}
