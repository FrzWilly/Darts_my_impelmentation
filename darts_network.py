import numpy as np
from torch import nn
import collections

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
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, gdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None, **kwargs):
        super(ConvDarts, self).__init__()
        self.lmda_cond = lmda_cond

        self.softmax = nn.Softmax(dim=0)

        #self.register_parameter('search', )
        self.search = True
        self.chosen_conv = -1
        self.chosen_acti = -1
        #self.module.register_parameter(self.search)
        #self.module.register_parameter(self.chosen)

        self.conv = ConvPoolBlock(in_channels, out_channels)
        self.acti = ActiBlock(in_channels, out_channels, igdn=False)

        # self.alphas = {
        #     'alpha_conv' : nn.parameter.Parameter(1e-3*torch.randn(5).cuda(), requires_grad=True),
        #     'alpha_acti' : nn.parameter.Parameter(1e-3*torch.randn(3).cuda(), requires_grad=True)
        # }

        self.register_parameter('alpha_conv', nn.parameter.Parameter(1e-3*torch.randn(5).cuda(), requires_grad=True))
        self.register_parameter('alpha_acti', nn.parameter.Parameter(1e-3*torch.randn(3).cuda(), requires_grad=True))

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        x = self.conv(inputs, self.softmax(self.alpha_conv), self.search, self.chosen_conv)
        x = self.acti(x, self.softmax(self.alpha_acti), self.search, self.chosen_acti)

        if self.lmda_cond:
            assert condition is not None
            x = self.condition(inputs, condition)

        return x

    def flops(self, input):
        total_flops = torch.Tensor([0])
        for layer in self.model:
            total_flops += layer.flops(input)
            input = layer(input)
        return total_flops


class TransposedConvDarts(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, igdn=True, relu=False, lmda_cond=False,
                 lmda_depth=None):
        super(TransposedConvDarts, self).__init__()
        self.lmda_cond = lmda_cond

        self.deconv = ConvTransBlock(in_channels, out_channels)
        self.acti = ActiBlock(in_channels, out_channels, igdn=igdn)

        self.softmax = nn.Softmax(dim=0)

        self.search = True
        self.chosen_deconv = -1
        self.chosen_acti = -1

        # self.alphas = {
        #     'alpha_deconv' : self.register_parameter('alpha_deconv', nn.parameter.Parameter(1e-3*torch.randn(5).cuda(), requires_grad=True)),
        #     'alpha_deconv' : self.register_parameter('alpha_acti', nn.parameter.Parameter(1e-3*torch.randn(3).cuda(), requires_grad=True))
        # }

        self.register_parameter('alpha_deconv', nn.parameter.Parameter(1e-3*torch.randn(5).cuda(), requires_grad=True))
        self.register_parameter('alpha_acti', nn.parameter.Parameter(1e-3*torch.randn(3).cuda(), requires_grad=True))

        if lmda_cond:
            assert lmda_depth is not None, "Please setup the info of lambda."

            self.condition = ConditionalLayer(out_channels, lmda_depth)

    def forward(self, inputs, condition=None):
        x = self.deconv(inputs, self.softmax(self.alpha_deconv), self.search, self.chosen_deconv)
        x = self.acti(x, self.softmax(self.alpha_acti), self.search, self.chosen_acti)

        if self.lmda_cond:
            assert condition is not None
            x = self.condition(inputs, condition)

        return x

# encoding block operations
class Conv3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, input):
        return self.model(input)
    
    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 3 * 3 self.in_channels * self.out_channels * height * width

class Conv5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2)

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 5 * 5 self.in_channels * self.out_channels * height * width

class Conv7x7(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Conv2d(in_channels, out_channels, 7, stride=2, padding=3)

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 7 * 7 self.in_channels * self.out_channels * height * width

def MaxPool3x3(in_channels, out_channels):
    return nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

def AvgPool3x3(in_channels, out_channels):
    return nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)

class ConvDep(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return (3 * 3 + self.out_channels) * self.in_channels * height * width

class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        # print(C_out)
        # assert C_out % 2 == 0
        # self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        # x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
        # out = self.bn(out)
        return out        

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return self.C_in * self.C_out * height * width
        

#decoding block operations

class TransConv3x3(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 3 * 3 self.in_channels * self.out_channels * height * width

class TransConv5x5(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1)

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 5 * 5 self.in_channels * self.out_channels * height * width

class TransConv7x7(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.ConvTranspose2d(in_channels, out_channels, 7, stride=2, padding=3, output_padding=1)

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return 7 * 7 self.in_channels * self.out_channels * height * width

class TransConvDep(nn.Module):

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, input):
        return self.model(input)

    def flops(self, input):
        height, width = input.size(2), input.size(3)
        return (3 * 3 + self.out_channels) * self.in_channels * height * width

class ConvPoolBlock(nn.Module):

    def __init__(self, in_channels, out_channels, search=True, chosen=-1):
        super(ConvPoolBlock, self).__init__()
        self.search = search
        self.chosen = chosen
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.op = nn.ModuleList()
        self.op.append(Conv7x7(in_channels, out_channels))
        self.op.append(Conv5x5(in_channels, out_channels))
        self.op.append(Conv3x3(in_channels, out_channels))
        self.op.append(ConvDep(in_channels, out_channels))

        # self.maxpool = MaxPool3x3(in_channels, out_channels)
        # self.avgpool = AvgPool3x3(in_channels, out_channels)
        self.op.append(FactorizedReduce(in_channels, out_channels))

    def forward(self, input, alpha, search, chosen):

        if search == True:
            # print("conv ", chosen)

            x = []
            for i in range(len(self.op)):
                x.append(self.op[i](input))

            # if self.in_channels == self.out_channels:
            #     x.append(self.maxpool(input))
            #     x.append(self.avgpool(input))

            for i in range(len(x)):
                x[i] = x[i] * alpha[i]
                # print(i, " ", x[i].size())

            output = x[0]
            for i in range(len(x)-1):
                output = output + x[i+1]
        else:
            # print("conv ", chosen)
            output = self.op[chosen](input)
            # print(output.size())
        
        return output
    
    def flops(self, input, alpha, search, chosen):
        total_flops = torch.Tensor([0])
        if search == True:
            for i in range(len(self.op)):
                total_flops += self.op[i].flops(input) * alpha[i]
        else:
             total_flops += self.op[chosen].flops(input)
        return total_flops

class ConvTransBlock(nn.Module):

    def __init__(self, in_channels, out_channels, search=True, chosen=-1):
        super(ConvTransBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.search = search
        self.chosen = chosen
        
        self.op = nn.ModuleList()
        self.op.append(TransConv7x7(in_channels, out_channels))
        self.op.append(TransConv5x5(in_channels, out_channels))
        self.op.append(TransConv3x3(in_channels, out_channels))
        self.op.append(TransConvDep(in_channels, out_channels))

        self.op.append(nn.Conv2d(in_channels, out_channels, 1))
        #self.zero = Zero(2)

    def forward(self, input, alpha, search, chosen):

        if search == True:
            # print("deconv ", chosen)
            x = []
            for i in range(len(self.op)-1):
                x.append(self.op[i](input))
            x.append(self.op[4](nn.functional.interpolate(input, mode="bilinear", scale_factor=2, align_corners=True)))

            for i in range(len(x)):
                x[i] = x[i] * alpha[i]

            output = x[0]
            for i in range(len(x)-1):
                output = output + x[i+1]
                # print(i, " ", x[i].size())
        
        else:
            # print("deconv ", chosen)
            output = self.op[chosen](input)
            # print(output.size())
        
        return output

    def flops(self, input, alpha, search, chosen):
        total_flops = torch.Tensor([0])
        if search == True:
            for i in range(len(self.op) - 1):
                total_flops += self.op[i].flops(input) * alpha[i]
            total_flops += self.op[4].flops(nn.functional.interpolate(input, mode="bilinear", scale_factor=2, align_corners=True)) * alpha[4]
        else:
             total_flops += self.op[chosen].flops(input)
        return total_flops
        
class ActiBlock(nn.Module):

    def __init__(self, in_channels, out_channels, search=True, chosen=-1, igdn=False):
        super(ActiBlock, self).__init__()
        self.search = search
        self.chosen = chosen
        self.relu = nn.ReLU(inplace=True)
        self.gdn = GeneralizedDivisiveNorm(out_channels, inverse=igdn)

    def forward(self, input, alpha, search, chosen):

        if search == True:
            # print("acti ", chosen)
            x = []
            x.append(self.relu(input)) 
            x.append(self.gdn(input))
            x.append(input)

            for i in range(len(x)):
                x[i] = x[i] * alpha[i]
                # print(i, " ", x[i].size())

            output = x[0]
            for i in range(len(x)-1):
                output = output + x[i+1]
        else:
            # print("acti ", chosen)
            if chosen == 0:
                output = self.relu(input)
            elif chosen == 1:
                output = self.gdn(input)
            else:
                output = input
            # print(output.size())
        
        return output

    def flops(self, input, alpha, search, chosen):
        total_flops = torch.Tensor([0])
        height, width = input.size(2), input.size(3)
        if search == True:
            x = []
            x.append(height * width)
            x.append(height * width * 2)
            x.append(0)
            for i in range(len(x)):
                total_flops += x[i] * alpha[i]
        else:
            if chosen == 0:
                total_flops += height * width
            elif chosen == 1:
                total_flops += height * width * 2    
            else:
                total_flops += 0
        return total_flops


class AnalysisTransform(nn.Module):
    """AnalysisTransform"""

    def __init__(self, num_filters, num_features, **kwargs):
        super(AnalysisTransform, self).__init__()

        self.model = nn.Sequential(collections.OrderedDict([
          ('l1', ConvDarts(3, num_filters)),
          ('l2', ConvDarts(num_filters, num_filters)),
          ('l3', ConvDarts(num_filters, num_filters)),
          ('l4', ConvDarts(num_filters, num_features))
        ]))

        # self.l1 = ConvDarts(3, num_filters, alpha_conv, alpha_acti)
        # self.l2 = ConvDarts(num_filters, num_filters, alpha_conv, alpha_acti)
        # self.l3 = ConvDarts(num_filters, num_filters, alpha_conv, alpha_acti)
        # self.l4 = ConvDarts(num_filters, num_features, alpha_conv, alpha_acti)

    def forward(self, inputs, condition=None):

        output = self.model(inputs)

        return output

    def flops(self, input):
        total_flops = torch.Tensor([0])
        for layer in self.model:
            total_flops += layer.flops(input)
            input = layer(input)
        return total_flops


class SynthesisTransform(nn.Module):
    """SynthesisTransform"""

    def __init__(self, num_filters, num_features, **kwargs):
        super(SynthesisTransform, self).__init__()

        self.model = nn.Sequential(collections.OrderedDict([
          ('d1', TransposedConvDarts(num_features, num_filters)),
          ('d2', TransposedConvDarts(num_filters, num_filters)),
          ('d3', TransposedConvDarts(num_filters, num_filters)),
          ('d4', TransposedConvDarts(num_filters, 3))
        ]))

        # self.d1 = TransposedConvDarts(num_features, num_filters, alpha_deconv, alpha_acti)
        # self.d2 = TransposedConvDarts(num_filters, num_filters, alpha_deconv, alpha_acti)
        # self.d3 = TransposedConvDarts(num_filters, num_filters, alpha_deconv, alpha_acti)
        # self.d4 = TransposedConvDarts(num_filters, 3, alpha_deconv, alpha_acti)

    def forward(self, inputs, condition=None):

        output = self.model(inputs)

        return output

    def flops(self, input):
        total_flops = torch.Tensor([0])
        for layer in self.model:
            total_flops += layer.flops(input)
            input = layer(input)
        return total_flops

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

        # self.alphas = {
        #     'alpha_conv' : nn.parameter.Parameter(1e-3*torch.randn(6).cuda(), requires_grad=True),
        #     'alpha_deconv' : nn.parameter.Parameter(1e-3*torch.randn(5).cuda(), requires_grad=True),
        #     'alpha_acti' : nn.parameter.Parameter(1e-3*torch.randn(3).cuda(), requires_grad=True)
        # }

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

    def arch_parameters(self):
        arch_p = []
        for v in self.analysis.model:
            arch_p.append(v.alpha_conv)
            arch_p.append(v.alpha_acti)
        for v in self.synthesis.model:
            arch_p.append(v.alpha_deconv)
            arch_p.append(v.alpha_acti)

        return arch_p

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

    def flops(self, input):
        features = self.analysis(input)
        hyperpriors = self.hyper_analysis(features)
        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)
        if self.args.Mean:
            sigma_mu = self.hyper_synthesis(z_tilde)
            sigma, mu = sigma_mu[:, :self.args.num_features], sigma_mu[:, self.args.num_features:]
            y_tilde, y_likelihood = self.conditional_bottleneck(features, sigma, mean=mu)
        else:
            sigma = self.hyper_synthesis(z_tilde)
            y_tilde, y_likelihood = self.conditional_bottleneck(features, sigma)

        total_flops  = self.analysis.flops(input)
        total_flops += self.synthesis.flops(y_tilde)
        return total_flops
