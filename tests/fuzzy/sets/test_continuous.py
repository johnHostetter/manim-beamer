"""
Test that various continuous fuzzy set implementations are working as intended, such as the
Gaussian fuzzy set (i.e., membership function), and the Triangular fuzzy set (i.e., membership
function).
"""
import unittest
import numpy as np

import torch

from soft.utilities.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian, Triangular  # pyTorch implementations


def gaussian_numpy(element: torch.Tensor, center: np.ndarray, sigma: np.ndarray):
    """
        Gaussian membership function that receives an 'element' value, and uses
        the 'center' and 'sigma' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Gaussian fuzzy set.
        sigma: The width of the Gaussian fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    return np.exp(-1.0 * (np.power(element - center, 2) / np.power(sigma, 2)))


def triangular_numpy(element: torch.Tensor, center: np.ndarray, width: np.ndarray):
    """
        Triangular membership function that receives an 'element' value, and uses
        the 'center' and 'width' to determine a degree of membership for 'element'.
        Implemented in Numpy and used in testing.

        https://www.mathworks.com/help/fuzzy/trimf.html

    Args:
        element: The element which we want to retrieve its membership degree.
        center: The center of the Triangular fuzzy set.
        width: The width of the Triangular fuzzy set.

    Returns:
        The membership degree of 'element'.
    """
    values = 1.0 - (1.0 / width) * np.abs(element - center)
    values[(values < 0)] = 0
    return values


class TestGaussian(unittest.TestCase):
    """
    Test the Gaussian fuzzy set (i.e., membership function).
    """

    def test_single_input(self) -> None:
        """
        Test that single input works for the Gaussian membership function.

        Returns:
            None
        """
        set_rng(0)
        element = 0.0
        n_inputs = 1
        gaussian_mf = Gaussian(n_inputs)
        sigma = gaussian_mf.sigmas.detach().numpy()
        center = gaussian_mf.centers.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(element))
        mu_numpy = gaussian_numpy(element, center, sigma)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.sigmas, torch.tensor(sigma)).all()
        assert torch.isclose(gaussian_mf.centers, torch.tensor(center)).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        gaussian_mf = Gaussian(in_features=elements.shape[1])
        centers, sigmas = (
            gaussian_mf.centers.detach().numpy(),
            gaussian_mf.sigmas.detach().numpy(),
        )
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.sigmas, torch.tensor(sigmas).float()).all()
        assert torch.isclose(gaussian_mf.centers, torch.tensor(centers).float()).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        gaussian_mf = Gaussian(in_features=centers.shape, centers=centers)
        sigmas = gaussian_mf.sigmas.detach().numpy()
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.sigmas, torch.tensor(sigmas).float()).all()
        assert torch.isclose(gaussian_mf.centers, torch.tensor(centers).float()).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

        expected_areas = torch.tensor(
            [0.7412324, 1.1474512, 0.13215375, 0.1972067, 0.45918167]
        )
        assert torch.isclose(gaussian_mf.area(), expected_areas).all()

    def test_multi_input_with_sigmas_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when sigmas are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        sigmas = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(in_features=elements.shape[1], widths=sigmas)
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(
            elements, gaussian_mf.centers.detach().numpy(), sigmas
        )

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.widths, sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Gaussian membership function when centers and
        sigmas are specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = torch.tensor([-0.5, -0.25, 0.25, 0.5, 0.75])
        sigmas = torch.tensor(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        gaussian_mf = Gaussian(
            in_features=elements.shape[1], centers=centers, widths=sigmas
        )
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers.detach().numpy(), sigmas)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.centers, centers).all()
        assert torch.isclose(gaussian_mf.widths, sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6
        ).all()

    def test_consistency(self) -> None:
        """
        Test that the results are consistent with the expected membership degrees.

        Returns:
            None
        """
        set_rng(0)
        element = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = torch.tensor(
            [
                [9.9984e-01, 4.3174e-01, 2.4384e-01, 1.1603e-02],
                [9.9992e-01, 4.2418e-01, 2.6132e-01, 7.6078e-04],
                [2.9000e-06, 9.5753e-01, 5.7272e-01, 1.2510e-01],
                [7.1018e-03, 9.9948e-01, 4.3918e-01, 7.5163e-03],
            ]
        )
        centers = torch.tensor(
            [
                [0.01497397, -1.3607662, 1.0883657, 1.9339248],
                [-0.01367673, 2.3560243, -1.8339163, -3.3379893],
                [-4.489564, -0.01467094, -0.13278057, 0.08638719],
                [0.17008819, 0.01596639, -1.7408595, 2.797653],
            ]
        )
        sigmas = torch.tensor(
            [
                [1.16553577, 1.48497267, 0.91602303, 0.91602303],
                [1.98733806, 2.53987592, 1.58646032, 1.24709336],
                [1.24709336, 0.10437003, 0.12908118, 0.08517358],
                [0.08517358, 1.54283158, 1.89779089, 1.27380911],
            ]
        )

        gaussian_mf = Gaussian(
            element.shape[1],
            centers=centers[: element.shape[1]],
            widths=sigmas[: element.shape[1]],
        )
        mu_pytorch = gaussian_mf(torch.tensor(element[0]))

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(gaussian_mf.centers, centers[: element.shape[1]]).all()
        assert torch.isclose(gaussian_mf.widths, sigmas[: element.shape[1]]).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.isclose(
            mu_pytorch.float(), target_membership_degrees, rtol=1e-1
        ).all()


class TestTriangular(unittest.TestCase):
    """
    Test the Triangular fuzzy set (i.e., membership function).
    """

    def test_single_input(self) -> None:
        """
        Test that single input works for the Triangular membership function.

        Returns:
            None
        """
        set_rng(0)
        element = 0.0
        n_inputs = 1
        triangular_mf = Triangular(n_inputs)
        center = triangular_mf.centers.detach().numpy()
        width = triangular_mf.widths.detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(element))
        mu_numpy = triangular_numpy(element, center, width)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(triangular_mf.centers, torch.tensor(center)).all()
        assert torch.isclose(triangular_mf.widths, torch.tensor(width)).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, atol=1e-2).all()

    def test_multi_input(self) -> None:
        """
        Test that multiple input works for the Triangular membership function.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        triangular_mf = Triangular(in_features=elements.shape[1])
        centers, widths = (
            triangular_mf.centers.detach().numpy(),
            triangular_mf.widths.detach().numpy(),
        )
        mu_pytorch = triangular_mf(elements)
        mu_numpy = triangular_numpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(triangular_mf.centers, torch.tensor(centers)).all()
        assert torch.isclose(triangular_mf.widths, torch.tensor(widths)).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, atol=1e-2
        ).all()

    def test_multi_input_with_centers_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        triangular_mf = Triangular(in_features=elements.shape[1], centers=centers)
        widths = triangular_mf.widths.detach().numpy()
        mu_pytorch = triangular_mf(elements)
        mu_numpy = triangular_numpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(triangular_mf.centers, torch.tensor(centers).float()).all()
        assert torch.isclose(triangular_mf.widths, torch.tensor(widths).float()).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, atol=1e-2
        ).all()

    def test_multi_input_with_widths_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when widths are
        specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        triangular_mf = Triangular(in_features=elements.shape[1], widths=widths)
        centers = triangular_mf.centers.detach().numpy()
        mu_pytorch = triangular_mf(elements)
        mu_numpy = triangular_numpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterward
        assert torch.isclose(triangular_mf.centers, torch.tensor(centers).float()).all()
        assert torch.isclose(triangular_mf.widths, torch.tensor(widths).float()).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, atol=1e-2
        ).all()

    def test_multi_input_with_both_given(self) -> None:
        """
        Test that multiple input works for the Triangular membership function when centers
        and widths are specified for the fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        elements = torch.tensor(
            [[0.41737163], [0.78705574], [0.40919196], [0.72005216]]
        )
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        widths = np.array(
            [0.1, 0.25, 0.5, 0.75, 1.0]
        )  # negative widths are missing sets
        triangular_mf = Triangular(
            in_features=elements.shape[1], centers=centers, widths=widths
        )
        mu_pytorch = triangular_mf(elements)
        mu_numpy = triangular_numpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterward
        assert (triangular_mf.centers.detach().numpy() == centers).all()
        assert np.isclose(triangular_mf.widths.detach().numpy(), widths).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(
            mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, atol=1e-2
        ).all()
