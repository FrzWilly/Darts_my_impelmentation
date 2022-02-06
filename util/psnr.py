import torch.nn as nn
import numpy as np
from math import log10


def PSNR(imgs1, imgs2, data_range=1.):
    mse = nn.functional.mse_loss(imgs1, imgs2)
    psnr = 20 * log10(data_range) - 10 * log10(mse.item())
    return psnr


def PSNR_np(imgs1, imgs2, data_range=255.):
    mse = np.mean(np.square(imgs1.astype(np.float) - imgs2.astype(np.float)))
    psnr = 20 * log10(data_range) - 10 * log10(mse.item())
    return psnr
