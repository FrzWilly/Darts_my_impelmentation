import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from entropy_models import EntropyBottleneck, GaussianConditional
from gdn import GeneralizedDivisiveNorm


# -------------------------------
#   Google ICLR'18 architecture (using OctConv)
# -------------------------------

class GOConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, block_stride=1, gdn=True, relu=False,
                 is_endpoint=False):

        assert not (relu and gdn), "Can't use GDN and ReLU at the same time."
        super(GOConv, self).__init__()

        def conv(in_channel, out_channel, k_size=5, stride=1, use_gdn=True, use_relu=False):
            if abs(stride) == 2:
                k_size -= 1

            m = nn.Sequential()

            if stride > 0:
                m.add_module('conv',
                             nn.Conv2d(in_channel, out_channel, k_size, stride, padding=(k_size - 1) // 2))
            else:
                m.add_module('deconv',
                             nn.ConvTranspose2d(in_channel, out_channel, k_size, -stride, padding=(k_size - 1) // 2))

            if use_gdn:
                m.add_module('gdn', GeneralizedDivisiveNorm(out_channel))

            if use_relu:
                m.add_module('relu', nn.ReLU(inplace=True))

            return m

        self.convHH = conv(in_channels, out_channels, kernel_size, stride=block_stride, use_gdn=gdn, use_relu=relu)
        self.convHL = conv(out_channels, out_channels, kernel_size, stride=2, use_gdn=gdn, use_relu=relu)
        self.convLL = conv(out_channels if is_endpoint else in_channels, out_channels, kernel_size, stride=block_stride,
                           use_gdn=gdn, use_relu=relu)

        if not is_endpoint:
            self.convLH = conv(out_channels, out_channels, kernel_size, stride=-2, use_gdn=gdn, use_relu=relu)

        self.is_endpoint = is_endpoint

    def forward(self, inputs_h, inputs_l=None):
        conv_hh = self.convHH(inputs_h)
        conv_hl = self.convHL(conv_hh)

        if self.is_endpoint:
            outputs_h, outputs_l = conv_hh, conv_hl
        else:
            assert inputs_l is not None

            conv_ll = self.convLL(inputs_l)
            conv_lh = self.convLH(conv_ll)

            outputs_h, outputs_l = conv_hh + conv_lh, conv_ll + conv_hl

        return outputs_h, outputs_l


class GOTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, block_stride=1, igdn=True, relu=False,
                 is_endpoint=False):

        assert not (relu and igdn), "Can't use GDN and ReLU at the same time."
        super(GOTConv, self).__init__()

        def deconv(in_channel, out_channel, k_size=5, stride=1, use_gdn=True, use_relu=False):
            if abs(stride) == 2:
                k_size -= 1

            m = nn.Sequential()
            if use_relu:
                m.add_module('relu', nn.ReLU(inplace=True))

            if use_gdn:
                m.add_module('igdn', GeneralizedDivisiveNorm(in_channel, inverse=True))

            if stride > 0:
                m.add_module('conv', nn.Conv2d(in_channel, out_channel, k_size, stride, padding=(k_size - 1) // 2))
            else:
                m.add_module('deconv',
                             nn.ConvTranspose2d(in_channel, out_channel, k_size, -stride, padding=(k_size - 1) // 2))
            return m

        self.convHH = deconv(in_channels, out_channels, kernel_size,
                             stride=-block_stride, use_gdn=igdn, use_relu=relu)

        if not is_endpoint:
            self.convHL = deconv(out_channels, out_channels, kernel_size, stride=2, use_gdn=igdn, use_relu=relu)

        self.convLL = deconv(in_channels, out_channels, kernel_size,
                             stride=-block_stride, use_gdn=igdn, use_relu=relu)
        self.convLH = deconv(out_channels, out_channels, kernel_size, stride=-2, use_gdn=igdn, use_relu=relu)

        self.is_endpoint = is_endpoint

    def forward(self, inputs_h, inputs_l):
        conv_hh = self.convHH(inputs_h)
        conv_ll = self.convLL(inputs_l)
        conv_lh = self.convLH(conv_ll)

        if self.is_endpoint:
            outputs_h, outputs_l = conv_hh + conv_lh, None
        else:
            outputs_h, outputs_l = conv_hh + conv_lh, conv_ll + self.convHL(conv_hh)

        return outputs_h, outputs_l


class OctAnalysisTransform(nn.Module):
    """AnalysisTransform"""

    def __init__(self, num_filters, num_features):
        super(OctAnalysisTransform, self).__init__()

        self.l1 = GOConv(3, num_filters, 5, is_endpoint=True)
        self.l2 = GOConv(num_filters, num_filters, 5, block_stride=2)
        self.l3 = GOConv(num_filters, num_filters, 5, block_stride=2)
        self.l4 = GOConv(num_filters, num_filters, 5, block_stride=2)
        self.l5 = GOConv(num_filters, num_features, 5, block_stride=2, gdn=False)

    def forward(self, inputs):
        conv1_h, conv1_l = self.l1(inputs)
        conv2_h, conv2_l = self.l2(conv1_h, conv1_l)
        conv3_h, conv3_l = self.l3(conv2_h, conv2_l)
        conv4_h, conv4_l = self.l4(conv3_h, conv3_l)
        conv5_h, conv5_l = self.l5(conv4_h, conv4_l)
        return conv5_h, conv5_l


class OctSynthesisTransform(nn.Module):
    """SynthesisTransform"""

    def __init__(self, num_filters, num_features):
        super(OctSynthesisTransform, self).__init__()

        self.d1 = GOTConv(num_features, num_filters, 5, block_stride=2)
        self.d2 = GOTConv(num_filters, num_filters, 5, block_stride=2)
        self.d3 = GOTConv(num_filters, num_filters, 5, block_stride=2)
        self.d4 = GOTConv(num_filters, num_filters, 5, block_stride=2)
        self.d5 = GOTConv(num_filters, 3, 5, is_endpoint=True, igdn=False)

    def forward(self, inputs_h, inputs_l):
        deconv1_h, deconv1_l = self.d1(inputs_h, inputs_l)
        deconv2_h, deconv2_l = self.d2(deconv1_h, deconv1_l)
        deconv3_h, deconv3_l = self.d3(deconv2_h, deconv2_l)
        deconv4_h, deconv4_l = self.d4(deconv3_h, deconv3_l)
        deconv5_h, _ = self.d5(deconv4_h, deconv4_l)

        return deconv5_h


class OctHyperAnalysisTransform(nn.Module):
    """HyperAnalysisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors):
        super(OctHyperAnalysisTransform, self).__init__()

        self.l1 = GOConv(num_features, num_filters, 3, gdn=False, relu=True)
        self.l2 = GOConv(num_filters, num_filters, 3, block_stride=2, gdn=False, relu=True)
        self.l3 = GOConv(num_filters, num_hyperpriors, 3, block_stride=2, gdn=False, relu=False)

    def forward(self, inputs_h, inputs_l):
        conv1_h, conv1_l = self.l1(inputs_h.abs(), inputs_l.abs())
        conv2_h, conv2_l = self.l2(conv1_h, conv1_l)
        conv3_h, conv3_l = self.l3(conv2_h, conv2_l)

        return conv3_h, conv3_l


class OctHyperSynthesisTransform(nn.Module):
    """HyperSynthesisTransform"""

    def __init__(self, num_filters, num_features, num_hyperpriors):
        super(OctHyperSynthesisTransform, self).__init__()

        self.d1 = GOTConv(num_hyperpriors, num_filters, 3, block_stride=2, igdn=False, relu=True)
        self.d2 = GOTConv(num_filters, num_filters, 3, block_stride=2, igdn=False, relu=True)
        self.d3 = GOTConv(num_filters, num_features, 3, igdn=False, relu=False)

    def forward(self, inputs_h, inputs_l):
        deconv1_h, deconv1_l = self.d1(inputs_h, inputs_l)
        deconv2_h, deconv2_l = self.d2(deconv1_h, deconv1_l)
        deconv3_h, deconv3_l = self.d3(deconv2_h, deconv2_l)

        return deconv3_h, deconv3_l


class OctGoogleHPCoder(nn.Module):
    """GoogleHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors):
        super(OctGoogleHPCoder, self).__init__()
        self.analysis = OctAnalysisTransform(num_filters, num_features)
        self.synthesis = OctSynthesisTransform(num_filters, num_features)
        self.hyper_analysis = OctHyperAnalysisTransform(
            num_filters, num_features, num_hyperpriors)
        self.hyper_synthesis = OctHyperSynthesisTransform(
            num_filters, num_features, num_hyperpriors)

        self.conditional_bn = GaussianConditional()
        self.entropy_bn_h = EntropyBottleneck(num_hyperpriors)
        self.entropy_bn_l = EntropyBottleneck(num_hyperpriors)

    @staticmethod
    def estimate_bpp(likelihood_list, num_pixels):
        return sum(likelihood.log().sum() for likelihood in likelihood_list) / (-np.log(2.) * num_pixels)

    def compress(self, input, filename):
        """Compresses an image."""
        features = self.analysis(input)
        hyperpriors = self.hyper_analysis(features)
        side_stream = self.entropy_bottleneck.compress(hyperpriors)
        stream = self.conditional_bottleneck.compress(features)

        return stream, side_stream, features.size(), hyperpriors.size()

    def decompress(self, stream):
        """Compresses an image."""
        device = next(self.parameters()).device
        stream, side_stream, y_shape, z_shape = stream
        z_hat = self.entropy_bottleneck.decompress(
            side_stream, z_shape, device=device)
        sigma = self.hyper_synthesis(z_hat)
        y_hat = self.conditional_bottleneck.decompress(
            stream, y_shape, sigma, device=device)
        reconstructed = self.synthesis(y_hat)

        return reconstructed

    def forward(self, input):
        features_h, features_l = self.analysis(input)
        hyperpriors_h, hyperpriors_l = self.hyper_analysis(features_h, features_l)

        z_h_tilde, z_h_likelihood = self.entropy_bn_h(hyperpriors_h)
        z_l_tilde, z_l_likelihood = self.entropy_bn_l(hyperpriors_l)

        sigma_h, sigma_l = self.hyper_synthesis(z_h_tilde, z_l_tilde)

        y_h_tilde, y_h_likelihood = self.conditional_bn(features_h, sigma_h)
        y_l_tilde, y_l_likelihood = self.conditional_bn(features_l, sigma_l)

        reconstructed = self.synthesis(y_h_tilde, y_l_tilde)

        return reconstructed, self.estimate_bpp([y_h_likelihood, y_l_likelihood, z_h_likelihood, z_l_likelihood],
                                                input.numel() / 3)
