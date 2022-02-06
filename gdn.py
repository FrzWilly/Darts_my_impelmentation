import torch
from torch.nn import Module, Parameter
from torch.nn import functional as F

from util.math import lowerbound


def generalized_divisive_norm(input, inverse: bool, gamma, beta, eps: float = 1e-5):
    """generalized divisive normalization"""
    assert gamma.size(0) == gamma.size(1), "gamma must be a square matrix"

    # Norm pool calc
    x = input.view(input.size()[:2] + (-1,))
    norm_pool = F.conv1d(x.pow(2), gamma.unsqueeze(-1), beta).add(eps).sqrt()

    # Apply norm
    if inverse:
        output = x * norm_pool
    else:
        output = x / norm_pool

    return output.view_as(input)


class NonnegativeParameterizer():
    """Object encapsulating nonnegative parameterization as needed for GDN.

    The variable is subjected to an invertible transformation that slows down the
    learning rate for small values.

    Args:
        offset: Offset added to the reparameterization of beta and gamma.
            The reparameterization of beta and gamma as their square roots lets the
            training slow down when their values are close to zero, which is desirable
            as small values in the denominator can lead to a situation where gradient
            noise on beta/gamma leads to extreme amounts of noise in the GDN
            activations. However, without the offset, we would get zero gradients if
            any elements of beta or gamma were exactly zero, and thus the training
            could get stuck. To prevent this, we add this small constant. The default
            value was empirically determined as a good starting point. Making it
            bigger potentially leads to more gradient noise on the activations, making
            it too small may lead to numerical precision issues.
    """

    def __init__(self, offset=2 ** -18):
        self.offset = offset
        self.pedestal = offset ** 2

    def init_(self, data):
        """no grad init data"""
        with torch.no_grad():
            pedestal = torch.full_like(data, self.pedestal)
            return torch.max(data.add_(pedestal), pedestal).sqrt_()

    def reparam(self, data, minmum=None):
        """reparam data"""
        if minmum is None:
            bound = self.offset
        else:
            bound = (minmum + self.pedestal) ** 0.5
        return lowerbound(data, bound).pow(2) - self.pedestal


class GeneralizedDivisiveNorm(Module):
    """Generalized divisive normalization layer.

    .. math::
        y[i] = x[i] / sqrt(sum_j(gamma[j, i] * x[j]^2) + beta[i])

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        inverse: If `False`, compute GDN response. If `True`, compute IGDN
            response (one step of fixed point iteration to invert GDN; the division is
            replaced by multiplication). Default: False.
        gamma_init: The gamma matrix will be initialized as the identity matrix
            multiplied with this value. If set to zero, the layer is effectively
            initialized to the identity operation, since beta is initialized as one. A
            good default setting is somewhere between 0 and 0.5.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.

    Shape:
        - Input: :math:`(B, C)`, `(B, C, L)`, `(B, C, H, W)` or `(B, C, D, H, W)`
        - Output: same as input

    Reference:
        paper: https://arxiv.org/abs/1511.06281
        github: https://github.com/tensorflow/compression/blob/master/tensorflow_compression/python/layers/gdn.py
    """
    _version = 4

    def __init__(self, num_features, inverse=False, gamma_init=.1, eps=1e-5):
        super(GeneralizedDivisiveNorm, self).__init__()
        self.num_features = num_features
        self.inverse = inverse
        self.gamma_init = gamma_init
        self.eps = eps

        self.weight = Parameter(torch.eye(num_features) * gamma_init)
        self.bias = Parameter(torch.ones(num_features))

        self.parameterizer = NonnegativeParameterizer()
        self.parameterizer.init_(self.weight)
        self.parameterizer.init_(self.bias)

    @property
    def gamma(self):
        return self.parameterizer.reparam(self.weight)

    @property
    def beta(self):
        return self.parameterizer.reparam(self.bias)

    def forward(self, input):
        return generalized_divisive_norm(input, self.inverse, self.gamma, self.beta, self.eps)

    def extra_repr(self):
        s = '{num_features}'
        if self.inverse:
            s += ', inverse=True'
        s += ', gamma_init={gamma_init}, eps={eps}'
        return s.format(**self.__dict__)
