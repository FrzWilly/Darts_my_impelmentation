"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.

--------------------------------------------------------------------------------

Very thin wrapper around torchac, for arithmetic coding.

"""
import torch
try:
    from torchac import torchac
except:
    print("Waring!! Can not import torchac.")


class ArithmeticCoder(object):
    def __init__(self, L=None):
        self.L = L
        self._cached_cdf = None

    def _check_cdf_shape(self, cdf):
        assert cdf.dim() == 4, cdf.shape
        if self.L is None:
            return
        Lp = cdf.size(-1)
        assert Lp == self.L + 1, (Lp, self.L)

    def range_encode(self, symbols, pmf=None, pmf_length=None):
        """
        :param symbols: symbols to encode, NHW
        :param pmf: pmf to use, NHWL
        :param pmf_length: pmf_length to use, NHW
        :return: symbols encode to a bytes string
        """
        assert symbols.dim() == 3, symbols.shape
        if symbols.dtype != torch.int16:
            raise TypeError(symbols.dtype)
        if pmf is None and self._cached_cdf is not None:
            cdf = self._cached_cdf
        else:
            cdf = pmf2cdf(pmf)
        self._check_cdf_shape(cdf)
        # if (symbols.max() - symbols.min()+1) > cdf.size(-1):
        #     raise IndexError('symbol out of range')

        pmf_length = pmf_length.to('cpu', non_blocking=True).reshape(-1).contiguous()
        symbols = symbols.to('cpu', non_blocking=True).reshape(-1).contiguous()

        out_bytes = torchac.encode_cdf(cdf, pmf_length, symbols)

        return out_bytes

    def range_decode(self, encoded_bytes, outbound_bytes, pmf, pmf_length):
        """
        :param encoded_bytes: bytes encoded by range_encode
        :param outbound_bytes: outbound bytes encoded by range_encode
        :param pmf: pmf to use, NHWL
        :param pmf_length: pmf_length to use, NHW
        :return: decoded matrix as np.int16, NHW
        """
        cdf = pmf2cdf(pmf)
        self._check_cdf_shape(cdf)

        pmf_length = pmf_length.to('cpu', non_blocking=True).reshape(-1).contiguous()

        decoded = torchac.decode_cdf(cdf, pmf_length, encoded_bytes, outbound_bytes)

        return decoded.reshape(*cdf.size()[:3])


def pmf2cdf(pmf):
    """
    :param pmf: NHWL
    :return: NHW(L+1) as int32 on CPU!
    """
    precision = 16

    cdf = pmf.cumsum(dim=-1, dtype=torch.float64).mul_(2**precision).clamp_max(2**precision - 1).round()
    cdf = torch.cat((torch.zeros_like(cdf[..., 0:1]), cdf), dim=-1)
    cdf = cdf.to('cpu', dtype=torch.int16, non_blocking=True)
    return cdf


def gen_uniform_pmf(size, L):
    N, _, H, W = size
    assert N == 1
    histo = torch.ones(L, dtype=torch.float32) / L
    assert (1 - histo.sum()).abs() < 1e-5, (1 - histo.sum()).abs()
    extendor = torch.ones(N, H, W, L)
    pmf = extendor * histo
    return pmf
