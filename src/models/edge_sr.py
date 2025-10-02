import torch
import torch.nn as nn
import torch.nn.functional as F

def sobel_edges_rgb(x):
    gray = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=x.device, dtype=x.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    return mag

class EdgeGuidedSR(nn.Module):
    def __init__(self, scale=3, in_ch_thermal=1, in_ch_optical=3, ch=32):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bicubic", align_corners=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 3, padding=1)
        )
    def forward(self, t_lr, o_hr):
        thermal_up = self.up(t_lr)
        edges = sobel_edges_rgb(o_hr)
        x = torch.cat([thermal_up, edges], dim=1)
        delta = self.fuse(x)
        return thermal_up + delta
