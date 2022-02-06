import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from coders import ArithmeticCoder
from util.universal_quant import uni_q

try:
    from .autograd import lowerbound
    from .functional import check_range
except:
    from util.math import lowerbound


class EntropyModel(nn.Module):
    """Entropy model (base class).

    Arguments:
      tail_mass: Float, between 0 and 1. The bottleneck layer automatically
        determines the range of input values based on their frequency of
        occurrence. Values occurring in the tails of the distributions will not
        be encoded with range coding, but using a Golomb-like code. `tail_mass`
        determines the amount of probability mass in the tails which will be
        Golomb-coded. For example, the default value of `2 ** -8` means that on
        average, one 256th of all values will use the Golomb code.
      likelihood_bound: Float. If positive, the returned likelihood values are
        ensured to be greater than or equal to this value. This prevents very
        large gradients with a typical entropy loss (defaults to 1e-9).
      range_coder_precision: Integer, between 1 and 16. The precision of the
        range coder used for compression and decompression. This trades off
        computation speed with compression efficiency, where 16 is the slowest
        but most efficient setting. Choosing lower values may increase the
        average codelength slightly compared to the estimated entropies.
    """

    def __init__(self, tail_mass=2 ** -8, likelihood_bound=1e-9, range_coder_precision=16, q_mode='noise'):
        super(EntropyModel, self).__init__()
        self.tail_mass = float(tail_mass)
        if not 0 < self.tail_mass < 1:
            raise ValueError(
                "`tail_mass` must be between 0 and 1, got {}.".format(self.tail_mass))
        self.likelihood_bound = float(likelihood_bound)
        self.range_coder_precision = int(range_coder_precision)
        self.q_mode = q_mode

    def quantize(self, input, mode):
        if self.training:
            return self._quantize(input, mode)
        else:
            return self._quantize(input, 'dequantize')

    def _quantize(self, input, mode):
        """Perturb or quantize a `Tensor` and optionally dequantize.

        Arguments:
        input: `Tensor`. The input values.
        mode: String. Can take on one of three values: `'noise'` (adds uniform
            noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
            (quantizes and produces integer symbols for range coder).

        Returns:
        The quantized/perturbed `input`. The returned `Tensor` should have type
        `self.dtype` if mode is `'noise'`, `'dequantize'`; `tf.int16` if mode is
        `'symbols'`.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    def dequantize(self, input):
        """Dequantize a `Tensor`.

        The opposite to `_quantize(input, mode='symbols')`.

        Arguments:
        input: `Tensor`. The range coder symbols.

        Returns:
        The dequantized `input`. The returned `Tensor` should have type
        `self.dtype`.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    def _likelihood(self, input):
        """Compute the likelihood of the input under the model.

        Arguments:
        input: `Tensor`. The input values.

        Returns:
        `Tensor` of same shape and type as `input`, giving the likelihoods
        evaluated at `input`.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    def get_pmf(self):
        """Compute the pmf of the model.

        Returns:
        `Tensor` of pmf.
        """
        raise NotImplementedError("Must inherit from EntropyModel.")

    def compress(self, input, scale=None, mean=None, return_sym=False):
        """Compress input and store their binary representations into strings.

        Arguments:
        input: `Tensor` with values to be compressed.

        Returns:
        compressed: String `Tensor` vector containing the compressed
            representation of each batch element of `input`.

        Raises:
        ValueError: if `input` has an integral or inconsistent `DType`, or
            inconsistent number of channels.
        """
        B, C, H, W = input.size()
        assert B == 1

        if scale is not None:
            self.scale = scale

        if mean is not None:
            self.mean = mean

        symbols = self._quantize(input, "symbols")  # CxHxW

        pmf, pmf_length, offset = self.get_pmf()  # CxHxWxL or Cx1x1xL
        assert symbols.dtype == pmf_length.dtype == offset.dtype == torch.int16

        symbols -= offset  # 1xCxHxW

        if pmf.size(1) == pmf.size(2) == 1:
            pmf = pmf.repeat(1, H, W, 1)
            pmf_length = pmf_length.repeat(1, 1, H, W)

        assert pmf.size() == (C, H, W, pmf.size(-1))  # CxHxWxL

        pmf = pmf.view(1, C, H*W, -1)
        symbols = symbols.view(1, C, H*W)
        pmf_length = pmf_length.view(1, C, H*W)

        ac = ArithmeticCoder(L=pmf.size(-1))

        if return_sym:
            truncated_symbols = symbols.view_as(input) + offset
            return ac.range_encode(symbols, pmf, pmf_length), self.dequantize(truncated_symbols.float().round())
        else:
            return ac.range_encode(symbols, pmf, pmf_length)

    def decompress(self, strings, outbound_strings, shape, scale=None, mean=None, device='cpu'):
        """Decompress values from their compressed string representations.

        Arguments:
        strings: A string `Tensor` vector containing the compressed data.

        Returns:
        The decompressed `Tensor`.
        """
        B, C, H, W = shape
        assert B == 1

        if scale is not None:
            self.scale = scale

        if mean is not None:
            self.mean = mean

        pmf, pmf_length, offset = self.get_pmf()  # CxHxWxL or Cx1x1xL
        assert pmf_length.dtype == offset.dtype == torch.int16

        if pmf.size(1) == pmf.size(2) == 1:
            pmf = pmf.repeat(1, H, W, 1)
            pmf_length = pmf_length.repeat(1, 1, H, W)

        assert pmf.size() == (C, H, W, pmf.size(-1))  # CxHxWxL

        pmf = pmf.view(1, C, H*W, -1)
        pmf_length = pmf_length.view(1, C, H*W)

        ac = ArithmeticCoder(L=pmf.size(-1))

        decoded_symbols = ac.range_decode(
            strings, outbound_strings, pmf, pmf_length)

        decoded_symbols = decoded_symbols.view(1, C, H, W).to(device=device)
        decoded_symbols += offset.to(device=device)

        return self.dequantize(decoded_symbols.float().round())

    def forward(self, input, scale=None, mean=None):
        self.scale = scale
        self.mean = mean

        output = self._quantize(
            input, self.q_mode if self.training else "dequantize")

        likelihood = self._likelihood(output)

        if self.likelihood_bound > 0:
            likelihood = lowerbound(likelihood, self.likelihood_bound)

        return output, likelihood


class FactorizeCell(nn.Module):
    def __init__(self, num_features, in_channel, out_channel, scale, factor=True):
        super(FactorizeCell, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale = scale

        self.weight = nn.Parameter(torch.Tensor(
            num_features, out_channel, in_channel))

        self.bias = nn.Parameter(torch.Tensor(
            num_features, out_channel, 1))

        if factor:
            self._factor = nn.Parameter(torch.Tensor(
                num_features, out_channel, 1))
        else:
            self.register_parameter('_factor', None)
        self.reset_parameters()

    def reset_parameters(self):
        init = np.log(np.expm1(1/self.scale/self.out_channel))
        nn.init.constant_(self.weight, init)
        nn.init.uniform_(self.bias, -0.5, 0.5)
        if self._factor is not None:
            nn.init.zeros_(self._factor)

    def extra_repr(self):
        s = '{in_channel}, {out_channel}'
        if self._factor is not None:
            s += ', factor=True'
        return s.format(**self.__dict__)

    def forward(self, input):
        output = F.softplus(self.weight) @ input + self.bias
        if self._factor is not None:
            output = output + torch.tanh(self._factor) * torch.tanh(output)
        return output


class EntropyBottleneck(EntropyModel):
    """Entropy bottleneck layer.

    This layer models the entropy of the tensor passing through it. During
    training, this can be used to impose a (soft) entropy constraint on its
    activations, limiting the amount of information flowing through the layer.
    After training, the layer can be used to compress any input tensor to a
    string, which may be written to a file, and to decompress a file which it
    previously generated back to a reconstructed tensor. The entropies estimated
    during training or evaluation are approximately equal to the average length of
    the strings in bits.

    The layer implements a flexible probability density model to estimate entropy
    of its input tensor, which is described in the appendix of the paper (please
    cite the paper if you use this code for scientific work):

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436

    The layer assumes that the input tensor is at least 2D, with a batch dimension
    at the beginning and a channel dimension as specified by `data_format`. The
    layer trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the input are i.i.d. (independent
    and identically distributed).

    Because data compression always involves discretization, the outputs of the
    layer are generally only approximations of its input. During training,
    discretization is modeled using additive uniform noise to ensure
    differentiability. The entropies computed during training are differential
    entropies. During evaluation, the data is actually quantized, and the
    entropies are discrete (Shannon entropies). To make sure the approximated
    tensor values are good enough for practical purposes, the training phase must
    be used to balance the quality of the approximation with the entropy, by
    adding an entropy term to the training loss. See the example in the package
    documentation to get started.

    Note: the layer always produces exactly one auxiliary loss and one update op,
    which are only significant for compression and decompression. To use the
    compression feature, the auxiliary loss must be minimized during or after
    training. After that, the update op must be executed at least once.
    """

    def __init__(self, num_features, init_scale=10, filters=(3, 3, 3), **kwargs):
        super(EntropyBottleneck, self).__init__(**kwargs)
        self.num_features = num_features
        self.init_scale = float(init_scale)
        self.filters = tuple(int(f) for f in filters)

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))

        self.factorizer = nn.Sequential()
        for i in range(len(self.filters) + 1):
            cell = FactorizeCell(num_features, filters[i], filters[i+1],
                                 scale, factor=i < len(self.filters))
            self.factorizer.add_module('l%d' % i, cell)

        # To figure out what range of the densities to sample, we need to compute
        # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. Since we
        # can't take inverses of the cumulative directly, we make it an optimization
        # problem:
        # `quantiles = argmin(|logit(cumulative) - target|)`
        # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
        # Taking the logit (inverse of sigmoid) of the cumulative makes the
        # representation of the right target more numerically stable.

        # Numerically stable way of computing logits of `tail_mass / 2`
        # and `1 - tail_mass / 2`.
        target = np.log(2 / self.tail_mass - 1)
        # Compute lower and upper tail quantile as well as median.
        self.target = torch.FloatTensor([-target, 0, target])

        quantiles = torch.FloatTensor(
            [[[-self.init_scale, 0, self.init_scale]]]).repeat(num_features, 1,  1)
        # self.register_buffer('quantiles', quantiles.requires_grad_())
        self.quantiles = nn.Parameter(quantiles)

        self._pmf = None
        self._pmf_length = None
        self._offset = None

    def extra_repr(self):
        return '(num_features): {num_features}'.format(**self.__dict__)

    def aux_loss(self):
        # for p in self.factorizer.parameters():
        #     p.requires_grad = False
        logits = self._logits_cumulative(self.quantiles)
        # for p in self.factorizer.parameters():
        #     p.requires_grad = True
        return torch.sum(torch.abs(logits - self.target.to(logits.device)))

    @property
    def medians(self):
        """Quantize such that the median coincides with the center of a bin."""
        return self.quantiles[:, 0, 1].detach()

    @torch.no_grad()
    def get_pmf(self):
        if not (self.training or self._pmf is None):  # use saved pmf
            return self._pmf, self._pmf_length, self._offset
        # Largest distance observed between lower tail quantile and median, and
        # between median and upper tail quantile.
        minima = self.medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int().relu_()

        maxima = self.quantiles[:, 0, 2] - self.medians
        maxima = torch.ceil(maxima).int().relu_()

        # PMF starting positions and lengths.
        pmf_start = self.medians - minima.float()
        pmf_length = maxima + minima + 1

        # Sample the densities in the computed ranges, possibly computing more
        # samples than necessary at the upper end.
        samples = torch.arange(torch.max(pmf_length), device=pmf_start.device)
        samples = samples + pmf_start.view(-1, 1, 1)

        # We strip the sigmoid from the end here, so we can use the special rule
        # below to only compute differences in the left tail of the sigmoid.
        # This increases numerical stability (see explanation in `call`).
        lower = self._logits_cumulative(samples - 0.5)
        upper = self._logits_cumulative(samples + 0.5)
        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) -
                        torch.sigmoid(sign * lower))

        for c in range(self.num_features):
            for idx in range(pmf_length[c].item()):
                pmf[c, 0, idx].clamp_min_(2 ** -15)

        pmf = F.normalize(pmf, p=1., dim=-1)

        # # Compute out-of-range (tail) masses.
        tail_mass = torch.sigmoid(
            lower[..., :1]) + torch.sigmoid(-upper[..., -1:])

        pmf = torch.cat([pmf, torch.zeros_like(tail_mass)], dim=-1)

        for idx in range(self.num_features):
            pmf[idx, 0, pmf_length[idx]] = tail_mass[idx, 0]

        self._pmf = pmf.unsqueeze(2)  # Cx1x1xL
        self._pmf_length = pmf_length.view(1, -1, 1, 1).short()
        self._offset = -minima.view(1, -1, 1, 1).short()

        return self._pmf, self._pmf_length, self._offset

    def _logits_cumulative(self, input):
        """Evaluate logits of the cumulative densities.

        Arguments:
        input: The values at which to evaluate the cumulative densities, expected
            to be a `Tensor` of shape `(channels, 1, batch)`.

        Returns:
        A `Tensor` of the same shape as `input`, containing the logits of the
        cumulative densities evaluated at the given input.
        """
        return self.factorizer(input)

        # # Convert to (channels, 1, batch) format by commuting channels to front
        # # and then collapsing.
        # x = input.transpose(0, 1)
        # C = input.size(1)
        # output = self.factorizer(x.reshape(C, 1, -1))

        # # Convert back to input tensor shape.
        # output = output.reshape_as(x).transpose(0, 1)
        # return output

    def _quantize(self, input, mode):
        """Perturb or quantize a `Tensor` and optionally dequantize.

        Arguments:
        input: `Tensor`. The input values.
        mode: String. Can take on one of three values: `'noise'` (adds uniform
            noise), `'dequantize'` (quantizes and dequantizes), and `'symbols'`
            (quantizes and produces integer symbols for range coder).

        Returns:
        The quantized/perturbed `input`. The returned `Tensor` should have type
        `self.dtype` if mode is `'noise'`, `'dequantize'`; `tf.int16` if mode is
        `'symbols'`.
        """
        # Add noise or quantize (and optionally dequantize in one step).
        if mode == "noise":
            noise = torch.rand_like(input) - 0.5
            return input + noise
        elif mode == "universal":
            return uni_q(input)

        medians = self.medians.view(-1, *(1,)*(input.dim()-2))
        outputs = torch.floor(input + (0.5 - medians))

        if mode == "dequantize":
            outputs = outputs.float()
            return outputs + medians
        else:
            assert mode == "symbols", mode
            outputs = outputs.short()
            return outputs

    def dequantize(self, input):
        """Dequantize a `Tensor`.

        The opposite to `_quantize(input, mode='symbols')`.

        Arguments:
        input: `Tensor`. The range coder symbols.

        Returns:
        The dequantized `input`. The returned `Tensor` should have type
        `self.dtype`.
        """
        medians = self.medians.view(-1, *(1,)*(input.dim()-2))
        outputs = input.float()
        return outputs + medians

    def _likelihood(self, input):
        """Compute the likelihood of the input under the model.

        Arguments:
        input: `Tensor`. The input values.

        Returns:
        `Tensor` of same shape and type as `input`, giving the likelihoods
        evaluated at `input`.
        """
        # Convert to (channels, 1, batch) format by commuting channels to front
        # and then collapsing.
        input = input.transpose(0, 1)
        C = input.size(0)
        x = input.reshape(C, 1, -1)

        # Evaluate densities.
        # We can use the special rule below to only compute differences in the left
        # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
        # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
        # with much higher precision than subtracting two numbers close to 1.
        lower = self._logits_cumulative(x - 0.5)
        upper = self._logits_cumulative(x + 0.5)
        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = -torch.sign(lower + upper).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) -
                               torch.sigmoid(sign * lower))

        # Convert back to input tensor shape.
        likelihood = likelihood.reshape_as(input).transpose(0, 1)
        return likelihood


class SymmetricConditional(EntropyModel):
    """Symmetric conditional entropy model (base class).

    Arguments:
      scale_bound: Float. Lower bound for scales. Any values in `scale` smaller
        than this value are set to this value to prevent non-positive scales. By
        default (or when set to `None`), uses the smallest value in
        `scale_table`. To disable, set to 0.
      mean: `Tensor`, the mean parameters for the conditional distributions. If
        `None`, the mean is assumed to be zero.
    """
    SCALES_MIN = 0.11
    SCALES_MAX = 256
    SCALES_LEVELS = 64

    def __init__(self, scale_bound=None, **kwargs):
        super(SymmetricConditional, self).__init__(**kwargs)

        # scale_table: Iterable of positive floats. For range coding, the scale
        # parameters in `scale` can't be used, because the probability tables need
        # to be constructed statically. Only the values given in this table will
        # actually be used for range coding. For each predicted scale, the next
        # greater entry in the table is selected. It's optimal to choose the
        # scales provided here in a logarithmic way.
        scale_table = torch.exp(torch.linspace(
            np.log(self.SCALES_MIN), np.log(self.SCALES_MAX), self.SCALES_LEVELS))
        # self.scale_table = tuple(sorted(float(s) for s in scale_table))
        self.scale_table = scale_table
        if self.scale_table.le(0).sum() > 0:
            raise ValueError(
                "`scale_table` must be an iterable of positive numbers.")
        self.mean = None

        self.scale_bound = self.SCALES_MIN if scale_bound is None else float(
            scale_bound)
        assert self.scale_bound >= 0
        self.distribution = self.get_distribution()

        t = torch.Tensor([self.tail_mass / 2])
        multiplier = -self._standardized_quantile(t)
        pmf_center = torch.ceil(self.scale_table * multiplier)

        self._offset = -pmf_center
        pmf_length = 2 * pmf_center.short() + 1
        self._pmf_length = pmf_length

        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        samples = torch.arange(torch.max(pmf_length))
        samples = torch.abs(samples-pmf_center.unsqueeze(1))
        samples_scale = self.scale_table.unsqueeze(1)
        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower  # shape(self.SCALES_LEVELS, 1479)

        for c in range(self.SCALES_LEVELS):
            for idx in range(pmf_length[c].item()):
                pmf[c, idx].clamp_min_(2 ** -15)

        pmf /= torch.sum(pmf, dim=-1, keepdim=True)

        # # Compute out-of-range (tail) masses.
        tail_mass = 2 * lower[:, :1]
        pmf = torch.cat([pmf, torch.zeros_like(tail_mass)], dim=-1)

        for idx in range(self.SCALES_LEVELS):
            pmf[idx, pmf_length[idx].item()] = tail_mass[idx]

        self._pmf = pmf

    def get_distribution(self):
        raise NotImplementedError("Must inherit from SymmetricConditional.")

    def resample(self, data, scale):
        return F.grid_sample(data, scale, mode='nearest', align_corners=True).permute(0, 2, 3, 1)

    @torch.no_grad()
    def get_pmf(self):
        scale = lowerbound(self.scale, self.scale_bound)
        pmf = self._pmf.t().unsqueeze(0).unsqueeze(2).to(scale.device)
        offset = self._offset.view(1, 1, 1, -1).to(scale.device)
        pmf_length = self._pmf_length.view(
            1, 1, 1, -1).to(scale.device, dtype=scale.dtype)

        B, C, H, W = scale.size()
        assert B == 1
        scale = scale.view(1, C, -1, 1)

        scale = (torch.log(scale) - np.log(self.SCALES_MIN)) / \
            (np.log(self.SCALES_MAX) - np.log(self.SCALES_MIN))
        flow = torch.cat([scale, torch.zeros_like(scale)], dim=-1)
        flow = flow * 2 - 1

        pmf = self.resample(pmf, flow).view(C, H, W, -1)
        offset = self.resample(offset, flow).view(1, C, H, W)
        pmf_length = self.resample(pmf_length, flow).view(1, C, H, W)

        return pmf, pmf_length.short(), offset.short()

    def _standardized_cumulative(self, input):
        """Evaluate the standardized cumulative density.

        Note: This function should be optimized to give the best possible numerical
        accuracy for negative input values.

        Arguments:
        input: `Tensor`. The values at which to evaluate the cumulative density.

        Returns:
        A `Tensor` of the same shape as `input`, containing the cumulative
        density evaluated at the given input.
        """
        raise NotImplementedError("Must inherit from SymmetricConditional.")

    def _standardized_quantile(self, quantile):
        """Evaluate the standardized quantile function.

        This returns the inverse of the standardized cumulative function for a
        scalar.

        Arguments:
        quantile: Float. The values at which to evaluate the quantile function.

        Returns:
        A float giving the inverse CDF value.
        """
        return self.distribution.icdf(quantile)

    def _quantize(self, input, mode):
        # Add noise or quantize (and optionally dequantize in one step).
        if mode == "noise":
            noise = torch.rand_like(input) - 0.5
            return input + noise
        elif mode == "universal":
            return uni_q(input)

        outputs = input
        if self.mean is not None:
            outputs = outputs - self.mean
        outputs = torch.floor(outputs + 0.5)

        if mode == "dequantize":
            if self.mean is not None:
                outputs = outputs + self.mean
            return outputs
        else:
            assert mode == "symbols", mode
            outputs = outputs.short()
            return outputs

    def dequantize(self, input):
        outputs = input.float()
        if self.mean is not None:
            outputs = outputs + self.mean
        return outputs

    def _likelihood(self, input):
        values = input
        if self.mean is not None:
            values = values - self.mean

        # This assumes that the standardized cumulative has the property
        # 1 - c(x) = c(-x), which means we can compute differences equivalently in
        # the left or right tail of the cumulative. The point is to only compute
        # differences in the left tail. This increases numerical stability: c(x) is
        # 1 for large x, 0 for small x. Subtracting two numbers close to 0 can be
        # done with much higher precision than subtracting two numbers close to 1.
        values = values.abs().neg()
        scale = lowerbound(self.scale, self.scale_bound)
        upper = self._standardized_cumulative((values + 0.5) / scale)
        lower = self._standardized_cumulative((values - 0.5) / scale)

        likelihood = upper - lower
        return likelihood


class GaussianConditional(SymmetricConditional):
    """Conditional Gaussian entropy model.

    The layer implements a conditionally Gaussian probability density model to
    estimate entropy of its input tensor, which is described in the paper (please
    cite the paper if you use this code for scientific work):

    > "Variational image compression with a scale hyperprior"<br />
    > J. Ballé, D. Minnen, S. Singh, S. J. Hwang, N. Johnston<br />
    > https://arxiv.org/abs/1802.01436
    """

    def get_distribution(self):
        return torch.distributions.normal.Normal(0., 1.)

    def _standardized_cumulative(self, input):
        # Using the complementary error function maximizes numerical precision.
        return 0.5 * torch.erfc(-(2 ** -0.5) * input)


class LogisticConditional(SymmetricConditional):
    """Conditional logistic entropy model.

    This is a conditionally Logistic entropy model, analogous to
    `GaussianConditional`.
    """

    def get_distribution(self):
        return torch.distributions.LogisticNormal(0., 1.)

    def _standardized_cumulative(self, input):
        return torch.sigmoid(input)


class LaplacianConditional(SymmetricConditional):
    """Conditional Laplacian entropy model.

    This is a conditionally Laplacian entropy model, analogous to
    `GaussianConditional`.
    """

    def get_distribution(self):
        return torch.distributions.Laplace(0., 1.)

    def _standardized_cumulative(self, input):
        exp = torch.exp(-torch.abs(input))
        return torch.where(input > 0, 2 - exp, exp) / 2
