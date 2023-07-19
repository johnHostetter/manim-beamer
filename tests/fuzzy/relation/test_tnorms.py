"""
Test various t-norm operations, such as the algebraic product.
"""
import unittest

import torch
import numpy as np

from soft.fuzzy.relation.tnorm import AlgebraicProduct


def algebraic_product(elements: np.ndarray, importance: np.ndarray) -> np.float32:
    """
    Numpy calculation of the algebraic product.

    Args:
        elements: The elements to be multiplied.
        importance: The importance of each element.

    Returns:
        The algebraic product of the given elements.
    """
    return np.product(elements * importance)


class TestAlgebraicProduct(unittest.TestCase):
    """
    Test the algebraic product operation.
    """

    def test_single_input(self) -> None:
        """
        The t-norm of a single input (w/o importance) should be == input.
        """
        element = torch.rand(1)
        n_inputs = 1
        tnorm = AlgebraicProduct(n_inputs)
        importance_before_calculation = tnorm.importance
        mu_pytorch = tnorm(element)
        mu_numpy = algebraic_product(
            element.detach().numpy(), importance_before_calculation.detach().numpy()
        )

        # make sure the parameters are still identical afterward
        assert torch.isclose(tnorm.importance, importance_before_calculation).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input(self) -> None:
        """
        Test that the algebraic product is correctly calculated when multiple inputs are given.

        Returns:
            None
        """
        elements = torch.rand(4)
        n_inputs = len(elements)
        tnorm = AlgebraicProduct(n_inputs)
        importance_before_calculation = tnorm.importance
        mu_pytorch = tnorm(elements)
        mu_numpy = algebraic_product(
            elements.detach().numpy(), importance_before_calculation.detach().numpy()
        )

        # make sure the parameters are still identical afterward
        assert torch.isclose(tnorm.importance, importance_before_calculation).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()

    def test_multi_input_with_importance_given(self) -> None:
        """
        Test that the algebraic product is correctly calculated when multiple inputs (and their
        varying degrees of importance) are given.

        Returns:
            None
        """
        elements = torch.rand(5)
        n_inputs = len(elements)
        importance_before_calculation = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        tnorm = AlgebraicProduct(n_inputs, importance=importance_before_calculation)
        mu_pytorch = tnorm(elements)
        mu_numpy = algebraic_product(
            elements.detach().numpy(), importance_before_calculation.detach().numpy()
        )

        # make sure the parameters are still identical afterward
        assert torch.isclose(tnorm.importance, importance_before_calculation).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-8).all()
