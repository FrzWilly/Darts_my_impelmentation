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


class ConvDarts(nn.Module):
    def __init__(self, in_channels, out_channels, alpha_conv, alpha_acti, kernel_size=5, stride=2, gdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None, **kwargs):
        super(ConvDarts, self).__init__()
        self.lmda_cond = lmda_cond

        self.conv = ConvPoolBlock(in_channels, out_channels)
        self.acti = ActiBlock(in_channels, out_channels)

        self.softmax = nn.Softmax(dim=0)

        self.alpha_conv = alpha_conv
        self.alpha_acti = alpha_acti

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        x = self.conv(inputs, self.softmax(self.alpha_conv))
        x = self.acti(x, self.softmax(self.alpha_acti))

        if self.lmda_cond:
            assert condition is not None
            x = self.condition(inputs, condition)

        return x


class TransposedConvDarts(nn.Module):
    def __init__(self, in_channels, out_channels, alpha_deconv, alpha_acti, kernel_size=5, stride=2, igdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None):
        super(TransposedConvDarts, self).__init__()
        self.lmda_cond = lmda_cond

        self.deconv = ConvTransBlock(in_channels, out_channels)
        self.acti = ActiBlock(in_channels, out_channels)

        self.softmax = nn.Softmax(dim=0)

        self.alpha_deconv = alpha_deconv
        self.alpha_acti = alpha_acti

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        x = self.deconv(inputs, self.softmax(self.alpha_deconv))
        x = self.acti(x, self.softmax(self.alpha_acti))

        if self.lmda_cond:
            assert condition is not None
            x = self.condition(inputs, condition)

        return x

# encoding block operations
def Conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

def Conv5x5(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2)

def Conv7x7(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3)

def MaxPool3x3(in_channels, out_channels):
    return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

def AvgPool3x3(in_channels, out_channels):
    return nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)
        
def ConvDep(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
    )

#decoding block operations
def TransConv3x3(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)

def TransConv5x5(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1)

def TransConv7x7(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, 7, stride=2, padding=3, output_padding=1)
        
def TransConvDep(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
    )

class ConvPoolBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvPoolBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv7x7 = Conv7x7(in_channels, out_channels)
        self.conv5x5 = Conv3x3(in_channels, out_channels)
        self.conv3x3 = Conv5x5(in_channels, out_channels)
        self.convDep = ConvDep(in_channels, out_channels)

        self.maxpool = MaxPool3x3(in_channels, out_channels)
        self.avgpool = AvgPool3x3(in_channels, out_channels)

    def forward(self, input, alpha):

        x = []
        x.append(self.conv7x7(input)) 
        x.append(self.conv5x5(input))
        x.append(self.conv3x3(input))
        x.append(self.convDep(input))
        if self.in_channels == self.out_channels:
            x.append(self.maxpool(input))
            x.append(self.avgpool(input))

        for i in range(len(x)):
            x[i] = x[i] * alpha[i]

        output = x[0]
        for i in range(len(x)-1):
            output = output + x[i+1]
        
        return output

class ConvTransBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvTransBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.tconv7x7 = TransConv7x7(in_channels, out_channels)
        self.tconv5x5 = TransConv3x3(in_channels, out_channels)
        self.tconv3x3 = TransConv5x5(in_channels, out_channels)
        self.tconvDep = TransConvDep(in_channels, out_channels)

    def forward(self, input, alpha):

        x = []
        x.append(self.tconv7x7(input)) 
        x.append(self.tconv5x5(input))
        x.append(self.tconv3x3(input))
        x.append(self.tconvDep(input))
        if self.in_channels == self.out_channels:
            x.append(nn.functional.interpolate(input, mode="bilinear", scale_factor=2, align_corners=True))

        for i in range(len(x)):
            x[i] = x[i] * alpha[i]

        output = x[0]
        for i in range(len(x)-1):
            output = output + x[i+1]
        
        return output
        
class ActiBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ActiBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gdn = GeneralizedDivisiveNorm(out_channels, inverse=True)

    def forward(self, input, alpha):

        x = []
        x.append(self.relu(input)) 
        x.append(self.gdn(input))
        x.append(input)

        for i in range(len(x)):
            x[i] = x[i] * alpha[i]

        output = x[0]
        for i in range(len(x)-1):
            output = output + x[i+1]
        
        return output



class AnalysisTransform(nn.Module):
    """AnalysisTransform"""

    def __init__(self, num_filters, num_features, alpha_conv, alpha_acti, **kwargs):
        super(AnalysisTransform, self).__init__()

        self.l1 = ConvDarts(3, num_filters, alpha_conv, alpha_acti)
        self.l2 = ConvDarts(num_filters, num_filters, alpha_conv, alpha_acti)
        self.l3 = ConvDarts(num_filters, num_filters, alpha_conv, alpha_acti)
        self.l4 = ConvDarts(num_filters, num_features, alpha_conv, alpha_acti)

    def forward(self, inputs, condition=None):
        conv1 = self.l1(inputs, condition)
        conv2 = self.l2(conv1, condition)
        conv3 = self.l3(conv2, condition)
        conv4 = self.l4(conv3, condition)
        return conv4


class SynthesisTransform(nn.Module):
    """SynthesisTransform"""

    def __init__(self, num_filters, num_features, alpha_deconv, alpha_acti, **kwargs):
        super(SynthesisTransform, self).__init__()

        self.d1 = TransposedConvDarts(num_features, num_filters, alpha_deconv, alpha_acti)
        self.d2 = TransposedConvDarts(num_filters, num_filters, alpha_deconv, alpha_acti)
        self.d3 = TransposedConvDarts(num_filters, num_filters, alpha_deconv, alpha_acti)
        self.d4 = TransposedConvDarts(num_filters, 3, alpha_deconv, alpha_acti)

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


class GoogleHyperPriorCoderDarts(nn.Module):
    """GoogleHyperPriorCoder"""
    def __init__(self, args):
        super(GoogleHyperPriorCoderDarts, self).__init__()
        self.args = args

        self.alphas = {
            'alpha_conv' : torch.autograd.Variable(1e-3*torch.randn(6).cuda(), requires_grad=True),
            'alpha_deconv' : torch.autograd.Variable(1e-3*torch.randn(5).cuda(), requires_grad=True),
            'alpha_acti' : torch.autograd.Variable(1e-3*torch.randn(3).cuda(), requires_grad=True)
        }

        self.analysis = AnalysisTransform(args.num_filters, args.num_features, alpha_conv=self.alphas['alpha_conv'], alpha_acti=self.alphas['alpha_acti'])
        self.synthesis = SynthesisTransform(args.num_filters, args.num_features, alpha_deconv=self.alphas['alpha_deconv'], alpha_acti=self.alphas['alpha_acti'])
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


