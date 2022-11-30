import torch
import unittest


class TestTSK(unittest.TestCase):
    def test_gradient_1(self):
        X = torch.tensor([[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]]).float()
        # first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(torch.tensor([[0, 1], [1, 2], [2, 3]]).float())
        actual_result = X.unsqueeze(dim=-1) - centers.T
        expected_result = torch.tensor([[[1.2000, 0.2000, -0.8000],
                                         [-0.8000, -1.8000, -2.8000]],
                                        [[1.1000, 0.1000, -0.9000],
                                         [-0.7000, -1.7000, -2.7000]],
                                        [[2.1000, 1.1000, 0.1000],
                                         [-0.9000, -1.9000, -2.9000]],
                                        [[2.7000, 1.7000, 0.7000],
                                         [-0.8500, -1.8500, -2.8500]],
                                        [[1.7000, 0.7000, -0.3000],
                                         [-0.7500, -1.7500, -2.7500]]])

        assert torch.isclose(actual_result, expected_result).all()

        actual_result = -1.0 * (torch.pow(actual_result, 2))

        sigmas = torch.nn.Parameter(torch.tensor([[0.0867, 0.3339],
                                                  [0.0518, 0.8080],
                                                  [0.0578, 0.3440]]))

    def test_gradient_2(self):
        a = torch.nn.Parameter(torch.tensor([0, 1]).float())
        b = torch.nn.Parameter(torch.tensor([1, 2]).float())
        c = 2 ** a
        assert c.grad_fn is not None
