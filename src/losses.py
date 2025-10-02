import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return F.l1_loss(pred, target)

def sobel_edges_1ch(x):
    kx = x.new_tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]])
    ky = x.new_tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]])
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return (gx*gx + gy*gy + 1e-6).sqrt()

def sobel_edges_rgb(x):
    gray = 0.2989 * x[:,0:1] + 0.5870 * x[:,1:2] + 0.1140 * x[:,2:3]
    return sobel_edges_1ch(gray)

def edge_alignment_loss(sr, o_hr):
    e_sr = sobel_edges_1ch(sr)
    e_opt = sobel_edges_rgb(o_hr)
    eps = 1e-6
    e_sr_n = e_sr / (e_sr.abs().mean(dim=(2,3), keepdim=True) + eps)
    e_opt_n = e_opt / (e_opt.abs().mean(dim=(2,3), keepdim=True) + eps)
    return (e_sr_n - e_opt_n).abs().mean()
