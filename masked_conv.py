import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.conv import _ConvNd

import torch
from torch import nn


class MaskedConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, mask, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask = mask
        super(MaskedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                           _pair(0), groups, bias, padding_mode)

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        weight = self.weight * self.mask
        return self._conv_forward(input, weight)


class MaskedCConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, lmda_depth, **kwargs):
        super(MaskedCConv, self).__init__()

        mask = torch.tensor([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0]], device='cuda:0')

        self.mcconv = MaskedConv2d(in_channels, out_channels, kernel_size, mask, **kwargs)

        self.lmda_depth = lmda_depth

        self.scale_layer = nn.Sequential(
            nn.Linear(lmda_depth, out_channels, bias=False),
            nn.Softplus()
        )
        self.bias_layer = nn.Linear(lmda_depth, out_channels, bias=False)

    def forward(self, inputs, lmda_idx):
        conv = self.mcconv(inputs)

        lmda_onehot = conv.new_zeros((lmda_idx.size()[0], self.lmda_depth)).scatter_(1, lmda_idx, 1)

        scale_factor = self.scale_layer(lmda_onehot)[:, :, None, None]
        bias_factor = self.bias_layer(lmda_onehot)[:, :, None, None]

        conv = scale_factor * conv + bias_factor

        return conv
