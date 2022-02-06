import torch
import unittest
from util import math
from torch.testing._internal.common_utils import TestCase


class MathTest(TestCase):
    def test_upper_bound(self):
        inputs_feed = torch.DoubleTensor([-1, 1])
        inputs_feed.requires_grad = True
        inputs_bound_feed = torch.zeros_like(inputs_feed)
        outputs_expected = torch.DoubleTensor([-1, 0])

        outputs = math.UpperBound.apply(inputs_feed, inputs_bound_feed)

        self.assertEqual(outputs, outputs_expected)

        outputs.backward(gradient=torch.ones_like(inputs_feed), retain_graph=True)

        self.assertEqual(inputs_feed.grad, torch.DoubleTensor([1, 1]))

        outputs.backward(gradient=-torch.ones_like(inputs_feed))

        self.assertEqual(inputs_feed.grad, torch.DoubleTensor([0, 1]))

    def test_lower_bound(self):
        inputs_feed = torch.DoubleTensor([-1, 1])
        inputs_feed.requires_grad = True
        inputs_bound_feed = torch.zeros_like(inputs_feed)
        outputs_expected = torch.DoubleTensor([0, 1])

        outputs = math.LowerBound.apply(inputs_feed, inputs_bound_feed)

        self.assertEqual(outputs, outputs_expected)

        outputs.backward(gradient=torch.ones_like(inputs_feed), retain_graph=True)

        self.assertEqual(inputs_feed.grad, torch.DoubleTensor([0, 1]))

        outputs.backward(gradient=-torch.ones_like(inputs_feed))

        self.assertEqual(inputs_feed.grad, torch.DoubleTensor([-1, 0]))


if __name__ == '__main__':
    TestCase.main()
