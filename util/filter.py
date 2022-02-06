"""Image Processing libary

James Chan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_gaussian_kernel(kernel_size, sigma):
    """Genarate 2D gaussian kernel"""
    x_grid, y_grid = torch.meshgrid(torch.arange(kernel_size),
                                    torch.arange(kernel_size))
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    g = (1./(2.*math.pi*variance)) * \
        (-((xy_grid - mean)**2).sum(-1)/(2*variance)).exp_()
    return g


class GaussianBlur0(nn.Conv2d):
    """GaussianBlur layer, a fixed layer to do Gaussian blur

    Args:
        channel (int): input channel
        window_size (int): must be odd number
        sigma (float or list): Standard deviation for Gaussian kernel
    """

    def __init__(self, channel, window_size=17, sigma=10):
        assert window_size % 2 == 1
        super().__init__(channel, channel, window_size,
                         padding=window_size//2, groups=channel, bias=False)

        self.filter_init(window_size, sigma)

    def filter_init(self, window_size, sigma):
        k = get_gaussian_kernel(window_size, sigma=sigma)
        for param in self.parameters():
            param.data.copy_(k)
            param.requires_grad = False


@torch.jit.script
def create_gauss_kernel1d(window_size: int, sigma: float, channel: int):
    """create 1D gauss kernel

    Args:
        window_size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
        channel (int): input channel
    """
    half_window = window_size // 2
    coords = torch.arange(-half_window, half_window+1).float()

    g = (-(coords ** 2) / (2 * sigma ** 2)).exp_()
    g.div_(g.sum())

    return g.reshape(1, 1, 1, -1).repeat(channel, 1, 1, 1)


@torch.jit.script
def gaussian_blur(x, window, use_padding: bool):
    """Blur input with 1-D gauss kernel

    Args:
        x (tensor): batch of tensors to be blured
        window (tensor): 1-D gauss kernel
        use_padding (bool): padding image before conv
    """
    C = x.size(1)
    padding = 0 if not use_padding else window.size(3) // 2
    out = F.conv2d(x, window, stride=1, padding=(0, padding), groups=C)
    out = F.conv2d(out, window.transpose(2, 3),
                   stride=1, padding=(padding, 0), groups=C)
    return out


class GaussianBlur(nn.Module):
    """GaussianBlur layer, a fixed layer to do Gaussian blur

    Args:
        channel (int): input channel
        window_size (int): must be odd number
        sigma (float or list): Standard deviation for Gaussian kernel
    """

    def __init__(self, channel, window_size=17, sigma=10):
        super().__init__()
        assert window_size % 2 == 1, 'Window size must be odd.'
        window = create_gauss_kernel1d(window_size, sigma, channel)
        self.register_buffer('window', window)
        self.kernel_size = window_size
        self.sigma = sigma

    def forward(self, input):
        return gaussian_blur(input, self.window, True)

    def extra_repr(self):
        return "kernel_size=({kernel_size}, {kernel_size}), sigma={sigma}".format(**self.__dict__)
