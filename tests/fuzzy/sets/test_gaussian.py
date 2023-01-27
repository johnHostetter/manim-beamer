import torch
import unittest
import numpy as np

from soft.fuzzy.sets.continuous import Gaussian  # a pyTorch implementation


def gaussian_numpy(x, center, sigma):
    """
        Gaussian membership function that receives an 'x' value, and uses the 'center' and 'sigma' to
        determine a degree of membership for 'x'. Implemented in Numpy and used in testing.

    Args:
        x: The element which we want to retrieve its membership degree.
        center: The center of the Gaussian fuzzy set.
        sigma: The width of the Gaussian fuzzy set.

    Returns:
        The membership degree of 'x'.
    """
    return np.exp(-1.0 * (np.power(x - center, 2) / np.power(sigma, 2)))


class TestGaussianMembershipFunction(unittest.TestCase):
    def test_single_input(self):
        element = 0.
        n_inputs = 1
        gaussian_mf = Gaussian(n_inputs)
        sigma = gaussian_mf.sigmas.detach().numpy()
        center = gaussian_mf.centers.detach().numpy()
        mu_pytorch = gaussian_mf(torch.tensor(element))
        mu_numpy = gaussian_numpy(element, center, sigma)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigma).all()
        assert (gaussian_mf.centers.detach().numpy() == center).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        gaussian_mf = Gaussian(in_features=elements.shape[1])
        centers, sigmas = gaussian_mf.centers.detach().numpy(), gaussian_mf.sigmas.detach().numpy()
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_centers_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        centers = np.array([0., 0.25, 0.5, 0.75, 1.0])
        gaussian_mf = Gaussian(in_features=elements.shape[1], centers=centers)
        sigmas = gaussian_mf.sigmas.detach().numpy()
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers, sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert (gaussian_mf.sigmas.detach().numpy() == sigmas).all()
        assert (gaussian_mf.centers.detach().numpy() == centers).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_sigmas_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        sigmas = torch.tensor([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        gaussian_mf = Gaussian(in_features=elements.shape[1], widths=sigmas)
        # we will now update the sigmas to be abs. value
        sigmas = torch.abs(sigmas)
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, gaussian_mf.centers.detach().numpy(), sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert torch.isclose(gaussian_mf.widths, sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_both_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        centers = torch.tensor([-0.5, -0.25, 0.25, 0.5, 0.75])
        sigmas = torch.tensor([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        gaussian_mf = Gaussian(in_features=elements.shape[1], centers=centers, widths=sigmas)
        # we will now update the sigmas to be abs. value
        sigmas = torch.abs(sigmas)
        mu_pytorch = gaussian_mf(elements)
        mu_numpy = gaussian_numpy(elements, centers.detach().numpy(), sigmas)

        # make sure the Gaussian parameters are still identical afterwards
        assert torch.isclose(gaussian_mf.centers, centers).all()
        assert torch.isclose(gaussian_mf.widths, sigmas).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_consistency(self):
        x = np.array([[0.0001712, 0.00393354, -0.03641258, -0.01936134]])
        target_membership_degrees = torch.tensor([[9.9984e-01, 4.3174e-01, 2.4384e-01, 1.1603e-02],
                                                  [9.9992e-01, 4.2418e-01, 2.6132e-01, 7.6078e-04],
                                                  [2.9000e-06, 9.5753e-01, 5.7272e-01, 1.2510e-01],
                                                  [7.1018e-03, 9.9948e-01, 4.3918e-01, 7.5163e-03]])
        centers = torch.tensor([[0.01497397, -1.3607662, 1.0883657, 1.9339248],
                                [-0.01367673, 2.3560243, -1.8339163, -3.3379893],
                                [-4.489564, -0.01467094, -0.13278057, 0.08638719],
                                [0.17008819, 0.01596639, -1.7408595, 2.797653]])
        sigmas = torch.tensor([[1.16553577, 1.48497267, 0.91602303, 0.91602303],
                               [1.98733806, 2.53987592, 1.58646032, 1.24709336],
                               [1.24709336, 0.10437003, 0.12908118, 0.08517358],
                               [0.08517358, 1.54283158, 1.89779089, 1.27380911]])

        gaussian_mf = Gaussian(x.shape[1], centers=centers[:x.shape[1]], widths=sigmas[:x.shape[1]])
        mu_pytorch = gaussian_mf(torch.tensor(x[0]))

        # make sure the Gaussian parameters are still identical afterwards
        assert torch.isclose(gaussian_mf.centers, centers[:x.shape[1]]).all()
        assert torch.isclose(gaussian_mf.widths, sigmas[:x.shape[1]]).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert torch.isclose(mu_pytorch.float(), target_membership_degrees, rtol=1e-1).all()
