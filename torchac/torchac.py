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
"""

# TODO some comments needed about [..., -1] == 0

import torch


# torchac can be built with and without CUDA support.
# Here, we try to import both torchac_backend_gpu and torchac_backend_cpu.
# If both fail, an exception is thrown here already.
#
# The right version is then picked in the functions below.
#
# NOTE:
# Without a clean build, multiple versions might be installed. You may use python setup.py clean --all to prevent this.
# But it should not be an issue.


import_errors = []


try:
    import torchac_backend_gpu
    CUDA_SUPPORTED = True
except ImportError as e:
    CUDA_SUPPORTED = False
    import_errors.append(e)

try:
    import torchac_backend_cpu
    CPU_SUPPORTED = True
except ImportError as e:
    CPU_SUPPORTED = False
    import_errors.append(e)


imported_at_least_one = CUDA_SUPPORTED or CPU_SUPPORTED


# if import_errors:
#     import_errors_str = '\n'.join(map(str, import_errors))
#     print(f'*** Import errors:\n{import_errors_str}')


if not imported_at_least_one:
    raise ImportError('*** Failed to import any torchac_backend! Make sure to install torchac with torchac/setup.py. '
                      'See the README for details.')


any_backend = torchac_backend_gpu if CUDA_SUPPORTED else torchac_backend_cpu


# print(f'*** torchac: GPU support: {CUDA_SUPPORTED} // CPU support: {CPU_SUPPORTED}')


def _get_gpu_backend():
    if not CUDA_SUPPORTED:
        raise ValueError('Got CUDA tensor, but torchac_backend_gpu is not available. '
                         'Compile torchac with CUDA support, or use CPU mode (see README).')
    return torchac_backend_gpu


def _get_cpu_backend():
    if not CPU_SUPPORTED:
        raise ValueError('Got CPU tensor, but torchac_backend_cpu is not available. '
                         'Compile torchac without CUDA support, or use GPU mode (see README).')
    return torchac_backend_cpu


def encode_cdf(cdf, pmf_length, symbols):
    """
    :param cdf: CDF as 1HWLp, as int32, on CPU!
    :param pmf_length: PMF length, as int16, on CPU!
    :param symbols: the symbols to encode, as int16, on CPU
    :return: byte-string, encoding `sym`
    """
    if cdf.is_cuda or symbols.is_cuda:
        raise ValueError('CDF and symbols must be on CPU for `encode_cdf`')
    # encode_cdf is defined in both backends, so doesn't matter which one we use!
    return any_backend.encode_cdf(cdf, pmf_length, symbols)


def decode_cdf(cdf, pmf_length, input_string, outbound_string):
    """
    :param cdf: CDF as 1HWLp, as int32, on CPU
    :param pmf_length: PMF length, as int16, on CPU
    :param input_string: byte-string, encoding some symbols `sym`.
    :param outbound_string: byte-string, encoding some outbound symbols in `sym`.
    :return: decoded `sym`.
    """
    if cdf.is_cuda:
        raise ValueError('CDF must be on CPU for `decode_cdf`')
    # encode_cdf is defined in both backends, so doesn't matter which one we use!
    return any_backend.decode_cdf(cdf, pmf_length, input_string, outbound_string)
