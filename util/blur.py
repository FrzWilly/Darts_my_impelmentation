import math

import torch
import torch.nn.functional as F
from torch import nn

from util.math import bound_tanh

SIGMAS = [-1, -0.5, 0, 0.5, 1]
SIGMAS = [item * 0 for item in SIGMAS]
# 0, 2, 4


def get_gaussian_kernel(kernel_size, sigma):
    """Genarate 2D gaussian kernel"""
    x_grid, y_grid = torch.meshgrid(torch.arange(kernel_size),
                                    torch.arange(kernel_size))
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    g = (1./(2.*math.pi*variance)) * \
        (-((xy_grid - mean)**2).sum(-1)/(2*variance)).exp()
    return g/(g.sum()+1e-12)


def make_scale_space(input, sigmas, kernel_sizes=[5, 3, 1, 3, 5]):
    """make scale_space volume"""
    volume = []
    C = input.size(1)
    for sigma, kernel_size in zip(sigmas, kernel_sizes):
        if sigma == 0:
            volume.append(input)
            continue
        kernel = get_gaussian_kernel(kernel_size, sigma).repeat(C, 1, 1, 1)
        padded = F.pad(input, pad=((kernel_size-1)//2, ) * 4,
                       mode='replicate')
        blured = F.conv2d(padded, kernel.to(input.device), groups=C)
        volume.append(blured)
    return torch.stack(volume, dim=2), sigmas


def scale_space(input, scale, sigmas):
    """scale_space"""
    volume = make_scale_space(input, sigmas)[0]

    B, C, D, H, W = volume.size()
    idx = (scale + 1).unsqueeze(2) * ((D-1) / 2)
    lb = idx.detach().floor().clamp(0, D-1)
    ub = (lb + 1).clamp(0, D-1)
    alpha = idx - idx.floor()

    lv = volume.gather(2, lb.long().expand(B, C, -1, H, W))
    uv = volume.gather(2, ub.long().expand(B, C, -1, H, W))

    val = (1-alpha) * lv + alpha * uv
    return val.squeeze(2), (1 - scale.abs()) * 2


class RaFC(nn.Module):

    def __init__(self):
        super(RaFC, self).__init__()

    def extra_repr(self):
        return f'mode={self.mode}'

    def forward(self, input, gamma):
        return scale_space(input, bound_tanh(gamma), SIGMAS)


class DropModel(nn.Module):
    """DropModel"""

    def __init__(self, num_features, multi_channel=False):
        super(DropModel, self).__init__()
        self.transform = nn.Conv2d(
            num_features, num_features if multi_channel else 1, 1, bias=False)

    def forward(self, input):
        conv = self.transform(input)
        return conv


class BlurBottleneck(nn.Module):
    def __init__(self, num_features):
        super(BlurBottleneck, self).__init__()

        self.drop_model = DropModel(num_features, multi_channel=True)
        self.rafc = RaFC()

    def forward(self, input):
        scale = self.drop_model(input)

        blurred_feature, level_map = self.rafc(input, scale)

        return blurred_feature, torch.mean(level_map, dim=[0, 1])
