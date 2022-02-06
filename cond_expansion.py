from network import *
from masked_conv import MaskedCConv
from skimage import io


class HyperPreprocessDecoder(nn.Module):
    def __init__(self, num_filters, num_hyperpriors, conditional=False, lmda_number=None):
        super(HyperPreprocessDecoder, self).__init__()

        self.d1 = TransposedConv(num_hyperpriors, num_filters, 5, stride=2, igdn=False, relu=True,
                                 lmda_cond=conditional, lmda_depth=lmda_number)
        self.d2 = TransposedConv(num_filters, num_filters, 5, stride=2, igdn=False, relu=True, lmda_cond=conditional,
                                 lmda_depth=lmda_number)
        self.d3 = TransposedConv(num_filters, num_filters, 3, stride=1, igdn=False, relu=True, lmda_cond=conditional,
                                 lmda_depth=lmda_number)

    def forward(self, inputs, lmda_idx=None):
        deconv1 = self.d1(inputs, lmda_idx)
        deconv2 = self.d2(deconv1, lmda_idx)
        deconv3 = self.d3(deconv2, lmda_idx)

        return deconv3


class HyperSynthesisExpand(nn.Module):
    def __init__(self, num_filters, num_features, conditional=False, lmda_number=None):
        super(HyperSynthesisExpand, self).__init__()
        self.num_features = num_features

        self.masked_conv = MaskedCConv(num_features, num_features*2, 5, lmda_depth=lmda_number, padding=2)

        self.d1 = TransposedConv(num_features * 4, num_filters * 3, 1, stride=1, igdn=False, relu=True,
                                 lmda_cond=conditional, lmda_depth=lmda_number)
        self.d2 = TransposedConv(num_filters * 3, num_filters * 2, 1, stride=1, igdn=False, relu=True,
                                 lmda_cond=conditional, lmda_depth=lmda_number)
        self.d3 = TransposedConv(num_filters * 2, num_features * 2, 1, stride=1, igdn=False, relu=False,
                                 lmda_cond=conditional, lmda_depth=lmda_number)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, features, lmda_idx):

        masked_feature = self.masked_conv(features, lmda_idx)

        inputs = torch.cat([masked_feature, inputs], dim=1)

        deconv1 = self.d1(inputs, lmda_idx)
        deconv2 = self.d2(deconv1, lmda_idx)
        deconv3 = self.d3(deconv2, lmda_idx)

        mean, var = deconv3[:, :self.num_features], self.relu(deconv3[:, self.num_features:])

        return mean, var


class SoftBinarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)

        return torch.round(inputs.clamp(0., 1.))

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors

        sig = torch.sigmoid((inputs - .5) * 20)
        der_sig = sig * (1 - sig)

        return grad_output * der_sig


class MaskGeneration(nn.Module):
    def __init__(self, num_features, lmda_depth):
        super(MaskGeneration, self).__init__()
        self.lmda_depth = lmda_depth

        threshold = torch.empty((lmda_depth, num_features)).fill_(0.0).to('cuda:0')
        self.threshold = nn.Parameter(threshold, requires_grad=True)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sigma, lmda_idx):
        lmda_onehot = sigma.new_zeros((lmda_idx.size()[0], self.lmda_depth)).scatter_(1, lmda_idx, 1)

        threshold = torch.matmul(lmda_onehot, self.threshold)[:, :, None, None]
        # TODO: add ReLU?

        if not self.training:
            print(threshold[:, 0, 0, 0])

        mask = SoftBinarize.apply(sigma - threshold)

        return mask


class ConditionalHPCoder(nn.Module):
    """SamsungHyperPriorCoder"""

    def __init__(self, args, lmda_number, q_mode='noise'):
        super(ConditionalHPCoder, self).__init__()
        self.q_mode = q_mode
        self.lmda_depth = lmda_number
        self.args = args

        self.analysis = AnalysisTransform(args.num_filters, args.num_features, conditional=True, lmda_number=lmda_number,
                                          prev_output=True)
        self.synthesis = SynthesisTransform(args.num_filters, args.num_features, conditional=True, lmda_number=lmda_number,
                                            hp_input=args.Bypass)
        self.hyper_analysis = HyperAnalysisTransform(
            args.num_filters, args.num_features, args.num_hyperpriors, conditional=True, lmda_number=lmda_number)
        self.hyper_synthesis = HyperSynthesisTransform(
            args.num_filters, args.num_features, args.num_hyperpriors, conditional=True, lmda_number=lmda_number,
            masked=args.Masked)

        if args.Masked:
            self.hyper_synthesis_expand = HyperSynthesisExpand(args.num_filters, args.num_features,
                                                               conditional=True, lmda_number=lmda_number)
        if args.Bypass:
            self.decoder_expand = HyperPreprocessDecoder(args.num_filters, args.num_hyperpriors, conditional=True,
                                                         lmda_number=lmda_number)

        self.mask_gen = MaskGeneration(args.num_features, lmda_number)

        self.conditional_bottleneck = GaussianConditional(q_mode=q_mode)
        self.entropy_bottleneck = EntropyBottleneck(args.num_hyperpriors, q_mode=q_mode)

        self.enhance = TucodecPostProcessing(num_features=32)

        if args.fix_base:
            self.analysis.requires_grad_(False)
            self.synthesis.requires_grad_(False)
            self.hyper_analysis.requires_grad_(False)
            self.hyper_synthesis.requires_grad_(False)
            self.conditional_bottleneck.requires_grad_(False)
            self.entropy_bottleneck.requires_grad_(False)

            if args.Masked:
                self.hyper_synthesis_expand.requires_grad_(False)
            if args.Bypass:
                self.decoder_expand.requires_grad_(False)

        if not args.train_threshold:
            self.mask_gen.threshold.requires_grad_(False)

    @staticmethod
    def estimate_bpp(y_entropy, z_entropy, num_pixels):
        return (y_entropy.view((y_entropy.shape[0], -1)).sum(dim=1) +
                z_entropy.view((z_entropy.shape[0], -1)).sum(dim=1)) / (-np.log(2.) * num_pixels)

    def compress(self, inputs, lmda_idx, q_size=1.):
        """Compresses an image."""
        features, prev_features = self.analysis(inputs, lmda_idx)
        hyperpriors = self.hyper_analysis(torch.cat([prev_features, features], dim=1), lmda_idx)

        side_stream, z_hat = self.entropy_bottleneck.compress(hyperpriors, return_sym=True)

        sigma = self.hyper_synthesis(z_hat, lmda_idx)

        features /= q_size

        # For y_tilde involved (Masked Hyper Decoder)
        if self.args.Masked:
            y_hat = self.conditional_bottleneck.quantize(features, "dequantize")
            mu, sigma = self.hyper_synthesis_expand(sigma, y_hat, lmda_idx)
            stream = self.conditional_bottleneck.compress(features, scale=sigma, mean=mu)
        else:
            stream = self.conditional_bottleneck.compress(features, scale=sigma)

        side_stream, side_outbound_stream = side_stream
        stream, outbound_stream = stream

        return [stream, side_outbound_stream, side_stream, outbound_stream], [features.size(), hyperpriors.size()]

    def decompress(self, stream_list, shape_list, lmda_idx=None, q_size=1.):
        """Compresses an image."""
        device = next(self.parameters()).device
        stream, side_outbound_stream, side_stream, outbound_stream = stream_list
        y_shape, z_shape = shape_list

        z_hat = self.entropy_bottleneck.decompress(side_stream, side_outbound_stream, z_shape, device=device)
        sigma = self.hyper_synthesis(z_hat, lmda_idx)

        y_hat = self.conditional_bottleneck.decompress(stream, outbound_stream, y_shape, scale=sigma, device=device)
        y_hat *= q_size

        if self.args.Bypass:
            post_z = self.decoder_expand(z_hat, lmda_idx)
            reconstructed = self.synthesis(torch.cat([post_z, y_hat], dim=1), lmda_idx)
        else:
            reconstructed = self.synthesis(y_hat, lmda_idx)

        return reconstructed

    def forward(self, inputs, lmda_idx):
        features, prev_features = self.analysis(inputs, lmda_idx)
        hyperpriors = self.hyper_analysis(torch.cat([prev_features, features], dim=1), lmda_idx)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyperpriors)

        sigma = self.hyper_synthesis(z_tilde, lmda_idx)

        if not self.training:
            q_size = 1.
        elif self.args.Q_size:
            q_size_exp = np.random.uniform(-1, 1)
            q_size = np.power(2., q_size_exp)
        else:
            q_size = 1.

        features /= q_size

        y_tilde = self.conditional_bottleneck.quantize(features, self.q_mode)

        # For y_tilde involved (Masked Hyper Decoder)
        if self.args.Masked:
            mu, sigma = self.hyper_synthesis_expand(sigma, y_tilde, lmda_idx)
            _, y_likelihood = self.conditional_bottleneck(features, scale=sigma, mean=mu)
        else:
            _, y_likelihood = self.conditional_bottleneck(features, scale=sigma)

        y_tilde *= q_size

        if self.args.imp_map:
            bin_mask = self.mask_gen(sigma, lmda_idx)
            y_tilde_masked = y_tilde * bin_mask
            y_entropy = y_likelihood.log() * bin_mask
            # if lmda_idx[0, 0].item() == 0 or lmda_idx[0, 0].item() == 2 or lmda_idx[0, 0].item() == 4:
            #     print("prepare plot")
            #     from matplotlib import pyplot as plt
            #
            #     config = ['high', 'mid', 'low']
            #     config = config[lmda_idx[0, 0].item() // 2]
            #
            #     lmda_onehot = sigma.new_zeros((lmda_idx.size()[0], self.lmda_depth)).scatter_(1, lmda_idx, 1)
            #     thresholds = torch.matmul(lmda_onehot, self.mask_gen.threshold)[0].cpu().numpy()
            #
            #     s_max = torch.max(torch.max(sigma, dim=2)[0], dim=2)[0][0].cpu().numpy()
            #     s_min = torch.min(torch.min(sigma, dim=2)[0], dim=2)[0][0].cpu().numpy()
            #     s_avg = torch.mean(sigma, dim=[2, 3])[0].cpu().numpy()
            #
            #     plt.plot(s_min, label='min $\sigma$', c='red')
            #     plt.plot(s_avg, label='mean $\sigma$', c='green')
            #     plt.plot(s_max, label='max $\sigma$', c='blue')
            #     plt.plot(thresholds, label='thresholds', c='black')
            #
            #     plt.legend()
            #
            #     plt.savefig("sigma_compare_" + config + ".png")
            #     plt.close()
            #
            #     for idx in range(bin_mask.size(1)):
            #         io.imsave("mask_" + config + "/" + config + "{:03d}.png".format(idx), bin_mask[0, idx].float().cpu().numpy())

        else:
            y_tilde_masked = y_tilde
            y_entropy = y_likelihood.log()

        if self.training and self.args.train_threshold:
            lmda_onehot = sigma.new_zeros((lmda_idx.size()[0], self.lmda_depth)).scatter_(1, lmda_idx, 1)
            thresholds = torch.matmul(lmda_onehot, self.mask_gen.threshold)

            channel_sigma = torch.mean(sigma, dim=[2, 3]).detach()

            assert thresholds.shape == channel_sigma.shape

            self.thresholds_loss = ((channel_sigma - thresholds) ** 2).mean()

        if self.args.Bypass:
            post_z = self.decoder_expand(z_tilde, lmda_idx)
            reconstructed = self.synthesis(torch.cat([post_z, y_tilde_masked], dim=1), lmda_idx)
        else:
            reconstructed = self.synthesis(y_tilde_masked, lmda_idx)

        enhanced = self.enhance(reconstructed, lmda_idx)

        return enhanced, self.estimate_bpp(y_entropy, z_likelihood.log(), inputs.shape[2] * inputs.shape[3])
