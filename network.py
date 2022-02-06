import numpy as np
from torch import nn

from entropy_models import EntropyBottleneck, GaussianConditional
from gdn import GeneralizedDivisiveNorm
from util.blur import BlurBottleneck
import torch


class ConditionalLayer(nn.Module):
    def __init__(self, num_features, num_condition):
        super(ConditionalLayer, self).__init__()
        self.scale_layer = nn.Sequential(
            nn.Linear(num_condition, num_features, bias=False),
            nn.Softplus()
        )
        self.bias_layer = nn.Linear(num_condition, num_features, bias=False)

    def forward(self, inputs, condition):
        scale_factor = self.scale_layer(condition).expand_as(inputs)
        bias_factor = self.bias_layer(condition).expand_as(inputs)

        return scale_factor * inputs + bias_factor

# -------------------------------
#   Google ICLR'18 architecture
# -------------------------------


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, gdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None, **kwargs):
        super(Conv, self).__init__()
        self.lmda_cond = lmda_cond

        self.m = nn.Sequential()

        self.m.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                            padding=(kernel_size - 1) // 2), **kwargs)
        if gdn:
            self.m.add_module('gdn', GeneralizedDivisiveNorm(out_channels))

        if relu:
            self.m.add_module('relu', nn.ReLU(inplace=True))

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        conv = self.m(inputs)

        if self.lmda_cond:
            assert condition is not None
            conv = self.condition(inputs, condition)

        return conv


class TransposedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, igdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None):
        super(TransposedConv, self).__init__()
        self.lmda_cond = lmda_cond

        self.m = nn.Sequential()

        self.m.add_module('deconv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                                       padding=(kernel_size - 1) // 2, output_padding=stride-1))

        if igdn:
            self.m.add_module('igdn', GeneralizedDivisiveNorm(out_channels, inverse=True))

        if relu:
            self.m.add_module('relu', nn.ReLU(inplace=True))

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        deconv = self.m(inputs)

        if self.lmda_cond:
            assert condition is not None
            deconv = self.condition(inputs, condition)

        return deconv


class AnalysisTransform(nn.Module):
    """AnalysisTransform"""

    def __init__(self, num_filters, num_features, **kwargs):
        super(AnalysisTransform, self).__init__()

        self.l1 = Conv(3, num_filters, 5, **kwargs)
        self.l2 = Conv(num_filters, num_filters, 5, **kwargs)
        self.l3 = Conv(num_filters, num_filters, 5, **kwargs)
        self.l4 = Conv(num_filters, num_features, 5, gdn=False, **kwargs)

    def forward(self, inputs, condition=None):
        conv1 = self.l1(inputs, condition)
        conv2 = self.l2(conv1, condition)
        conv3 = self.l3(conv2, condition)
        conv4 = self.l4(conv3, condition)
        return conv4


class SynthesisTransform(nn.Module):
    """SynthesisTransform"""

    def __init__(self, num_filters, num_features, **kwargs):
        super(SynthesisTransform, self).__init__()

        self.d1 = TransposedConv(num_features, num_filters, 5, **kwargs)
        self.d2 = TransposedConv(num_filters, num_filters, 5, **kwargs)
        self.d3 = TransposedConv(num_filters, num_filters, 5, **kwargs)
        self.d4 = TransposedConv(num_filters, 3, 5, igdn=False, **kwargs)

    def forward(self, inputs, condition=None):
        deconv1 = self.d1(inputs, condition)
        deconv2 = self.d2(deconv1, condition)
        deconv3 = self.d3(deconv2, condition)
        deconv4 = self.d4(deconv3, condition)
        return deconv4


class HyperAnalysisTransform(nn.Module):
    """HyperAnalysisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors, **kwargs):
        super(HyperAnalysisTransform, self).__init__()

        self.l1 = Conv(num_features, num_filters, 3, stride=1,
                       gdn=False, relu=True, **kwargs)
        self.l2 = Conv(num_filters, num_filters, 3, stride=2,
                       gdn=False, relu=True, **kwargs)
        self.l3 = Conv(num_filters, num_hyperpriors, 3,
                       stride=2, gdn=False, relu=False, **kwargs)

    def forward(self, inputs, condition=None):
        conv1 = self.l1(inputs.abs(), condition)
        conv2 = self.l2(conv1, condition)
        conv3 = self.l3(conv2, condition)
        return conv3


class HyperSynthesisTransform(nn.Module):
    """HyperSynthesisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors, **kwargs):
        super(HyperSynthesisTransform, self).__init__()

        self.d1 = TransposedConv(
            num_hyperpriors, num_filters, 3, stride=2, igdn=False, relu=True, **kwargs)
        self.d2 = TransposedConv(
            num_filters, num_filters, 3, stride=2, igdn=False, relu=True, **kwargs)
        self.d3 = TransposedConv(
            num_filters, num_features, 3, stride=1, igdn=False, relu=False, **kwargs)

    def forward(self, inputs, condition=None):
        deconv1 = self.d1(inputs, condition)
        deconv2 = self.d2(deconv1, condition)
        deconv3 = self.d3(deconv2, condition)
        return deconv3

def morphize_forward_hook(module, input, output):
    return output * module.gate

morph_target = [
    'analysis.l1.m.conv',
    'analysis.l2.m.conv',
    'analysis.l3.m.conv',
    'analysis.l4.m.conv',
    'synthesis.d1.m.deconv',
    'synthesis.d2.m.deconv',
    'synthesis.d3.m.deconv',
    'synthesis.d4.m.deconv'
]

class GoogleHyperPriorCoder(nn.Module):
    """GoogleHyperPriorCoder"""

    """ Morph Begin """
    def morph_loss(self):
        cnt, res = 0, 0
        for name, module in self.named_modules():
            if name not in morph_target:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                res += torch.abs(module.gate).sum()
                cnt += module.gate.size(1)
        return res / cnt

    def morphize(self):
        for name, module in self.named_modules():
            if name not in morph_target:
                continue
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.name = name
                print ("morph ", module.name)
                module.gate = nn.Parameter(torch.ones((1, module.out_channels, 1, 1)))
                module.handler = module.register_forward_hook(morphize_forward_hook)

    def morph_status(self):
        tot, cnt = 0, 0
        for name, module in self.named_modules():
            if name not in morph_target:
                continue
            if not isinstance(module, nn.Conv2d) and not isinstance(module, nn.ConvTranspose2d):
                continue
            weight = module.gate.data.reshape(-1)
            print (name, weight[weight > 0.01].size(0), '/', weight.size(0))
            tot += weight.size(0)
            cnt += weight[weight < 0.01].size(0)
        print (tot - cnt, '/', tot)

    def demorphize(self):
        tot, cnt = 0, 0
        for name, module in self.named_modules():
            if name not in morph_target:
                continue
            if not isinstance(module, nn.Conv2d) and not isinstance(module, nn.ConvTranspose2d):
                continue
            module.handler.remove()
            weight = module.gate.data.reshape(-1)
            print (name, weight[weight > 0.01].size(0), '/', weight.size(0))
            tot += weight.size(0)
            cnt += weight[weight < 0.01].size(0)
            weight[weight < 0.01] = 0
            if isinstance(module, nn.Conv2d):
                weight = weight.reshape(-1, 1, 1, 1)
            if isinstance(module, nn.ConvTranspose2d):
                weight = weight.reshape(1, -1, 1, 1)
            module.weight = nn.Parameter(module.weight * weight)
            if module.bias is not None:
                weight = weight.reshape(-1)
                module.bias = nn.Parameter(module.bias * weight)
        print (tot - cnt, '/', tot)
    """ Morph  End  """

    def __init__(self, args):
        super(GoogleHyperPriorCoder, self).__init__()
        self.args = args

        self.analysis = AnalysisTransform(args.num_filters, args.num_features)
        self.synthesis = SynthesisTransform(args.num_filters, args.num_features)
        self.hyper_analysis = HyperAnalysisTransform(
            args.num_filters, args.num_features, args.num_hyperpriors)
        self.hyper_synthesis = HyperSynthesisTransform(
            args.num_filters, args.num_features * 2 if args.Mean else args.num_features, args.num_hyperpriors)

        self.conditional_bottleneck = GaussianConditional()
        self.entropy_bottleneck = EntropyBottleneck(args.num_hyperpriors)

        if self.args.Blur:
            self.blur_bottleneck = BlurBottleneck(args.num_features)

        if args.command == "train" and args.fixed_enc:
            self.analysis.requires_grad_(False)

        if args.command == "train" and args.fixed_base:
            self.analysis.requires_grad_(False)
            self.synthesis.requires_grad_(False)
            self.hyper_analysis.requires_grad_(False)
            self.hyper_synthesis.requires_grad_(False)
            self.conditional_bottleneck.requires_grad_(False)
            self.entropy_bottleneck.requires_grad_(False)

    @staticmethod
    def estimate_bpp(likelihood, num_pixels):
        return likelihood.log().flatten(1).sum(1) / (-np.log(2.) * num_pixels)

    def compress(self, input):
        """Compresses an image."""
        features = self.analysis(input)
        if self.args.Blur:
            features, blur_scale = self.blur_bottleneck(features)

        hyperpriors = self.hyper_analysis(features)

        side_stream, z_hat = self.entropy_bottleneck.compress(hyperpriors, return_sym=True)
        side_stream, side_outbound_stream = side_stream

        if self.args.Mean:
            sigma_mu = self.hyper_synthesis(z_hat)
            sigma, mu = sigma_mu[:, :self.args.num_features], sigma_mu[:, self.args.num_features:]
            stream, y_hat = self.conditional_bottleneck.compress(features, scale=sigma, mean=mu, return_sym=True)
        else:
            sigma = self.hyper_synthesis(z_hat)
            stream, y_hat = self.conditional_bottleneck.compress(features, scale=sigma, return_sym=True)

        stream, outbound_stream = stream

        return [stream, outbound_stream, side_stream, side_outbound_stream], \
               [features.size(), hyperpriors.size()]

    def decompress(self, stream_list, shape_list):
        """Compresses an image."""
        device = next(self.parameters()).device
        stream, outbound_stream, side_stream, side_outbound_stream = stream_list
        y_shape, z_shape = shape_list

        z_hat = self.entropy_bottleneck.decompress(side_stream, side_outbound_stream, z_shape, device=device)

        if self.args.Mean:
            sigma_mu = self.hyper_synthesis(z_hat)
            sigma, mu = sigma_mu[:, :self.args.num_features], sigma_mu[:, self.args.num_features:]
            y_hat = self.conditional_bottleneck.decompress(stream, outbound_stream, y_shape, scale=sigma, mean=mu,
                                                           device=device)
        else:
            sigma = self.hyper_synthesis(z_hat)
            y_hat = self.conditional_bottleneck.decompress(stream, outbound_stream, y_shape, scale=sigma, device=device)

        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features = self.analysis(input)
        if self.args.Blur:
            features, _ = self.blur_bottleneck(features)

        hyperpriors = self.hyper_analysis(features)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        if self.args.Mean:
            sigma_mu = self.hyper_synthesis(z_tilde)
            sigma, mu = sigma_mu[:, :self.args.num_features], sigma_mu[:, self.args.num_features:]
            y_tilde, y_likelihood = self.conditional_bottleneck(features, sigma, mean=mu)
        else:
            sigma = self.hyper_synthesis(z_tilde)
            y_tilde, y_likelihood = self.conditional_bottleneck(features, sigma)

        reconstructed = self.synthesis(y_tilde)

        num_pixels = input.size(2) * input.size(3)
        y_rate = self.estimate_bpp(y_likelihood, num_pixels)
        z_rate = self.estimate_bpp(z_likelihood, num_pixels)
        return reconstructed, y_rate + z_rate


