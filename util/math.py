import torch


class UpperBound(torch.autograd.Function):
    """
    Same as `torch.min`, but with helpful gradient for `inputs > bound`.
    """

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_max(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input <= ctx.bound) | (grad_output > 0)
        return grad_output * pass_through, None


def upperbound(input, bound: float):
    return UpperBound.apply(input, bound)


class LowerBound(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, bound: float):
        ctx.save_for_backward(input)
        ctx.bound = bound

        return input.clamp_min(bound)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        pass_through = (input >= ctx.bound) | (grad_output < 0)
        return grad_output * pass_through.float(), None


def lowerbound(input, bound: float):
    return LowerBound.apply(input, bound)


def bound(input, min, max):
    """bound"""
    return upperbound(lowerbound(input, min), max)


def bound_sigmoid(input, scale=10):
    """bound_sigmoid"""
    return bound(input, -scale, scale).sigmoid()


def bound_tanh(input, scale=3):
    """bound_tanh"""
    return bound(input, -scale, scale).tanh()
