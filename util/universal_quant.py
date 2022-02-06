import torch


class UniversalQuantization(torch.autograd.Function):
    """
    Same as `torch.min`, but with helpful gradient for `inputs > bound`.
    """
    @staticmethod
    def forward(ctx, inputs):
        noise = torch.rand_like(inputs) - 0.5

        return torch.round(inputs + noise) - noise

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def uni_q(inputs):
    return UniversalQuantization.apply(inputs)
