import torch
import unittest
import numpy as np
from soft.fuzzy.relation.tnorm import AlgebraicProduct


def algebraic_product(elements, importance):
    return np.product(elements * importance)


class TestAlgebraicProduct(unittest.TestCase):
    def test_single_input(self):
        """
        The t-norm of a single input (w/o importance) should be == input.
        """
        element = torch.rand(1)
        n_inputs = 1
        tnorm = AlgebraicProduct(n_inputs)
        importance = tnorm.importance.detach().numpy()
        mu_pytorch = tnorm(element)
        mu_numpy = algebraic_product(element.detach().numpy(), importance)

        # make sure the parameters are still identical afterwards
        assert (tnorm.importance.detach().numpy() == importance).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input(self):
        elements = torch.rand(4)
        n_inputs = len(elements)
        tnorm = AlgebraicProduct(n_inputs)
        importance = tnorm.importance.detach().numpy()
        mu_pytorch = tnorm(elements)
        mu_numpy = algebraic_product(elements.detach().numpy(), importance)

        # make sure the parameters are still identical afterwards
        assert (tnorm.importance.detach().numpy() == importance).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input_with_importance_given(self):
        elements = torch.rand(5)
        n_inputs = len(elements)
        importance = np.array([0., 0.25, 0.5, 0.75, 1.0])
        tnorm = AlgebraicProduct(n_inputs, importance=importance)
        mu_pytorch = tnorm(elements)
        mu_numpy = algebraic_product(elements.detach().numpy(), importance)

        # make sure the parameters are still identical afterwards
        assert (tnorm.importance.detach().numpy() == importance).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()
