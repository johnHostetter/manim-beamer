import torch
import unittest
import numpy as np

from soft.fuzzy.sets import Triangular  # a pyTorch implementation
from soft.fuzzy.membership import triangular as TriangularNumpy  # a Numpy implementation


class TestTriangularMembershipFunction(unittest.TestCase):
    def test_single_input(self):
        element = 0.
        n_inputs = 1
        triangular_mf = Triangular(n_inputs)
        center = triangular_mf.centers.detach().numpy()
        width = triangular_mf.widths.detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(element))
        mu_numpy = TriangularNumpy(element, center, width)

        # make sure the Gaussian parameters are still identical afterwards
        assert (triangular_mf.centers.detach().numpy() == center).all()
        assert (triangular_mf.widths.detach().numpy() == width).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        triangular_mf = Triangular(in_features=elements.shape[1])
        centers, widths = triangular_mf.centers.detach().numpy(), triangular_mf.widths.detach().numpy()
        mu_pytorch = triangular_mf(elements)
        mu_numpy = TriangularNumpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterwards
        assert (triangular_mf.centers.detach().numpy() == centers).all()
        assert (triangular_mf.widths.detach().numpy() == widths).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_centers_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        centers = np.array([0., 0.25, 0.5, 0.75, 1.0])
        triangular_mf = Triangular(in_features=elements.shape[1], centers=centers)
        widths = triangular_mf.widths.detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(elements))
        mu_numpy = TriangularNumpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterwards
        assert (triangular_mf.centers.detach().numpy() == centers).all()
        assert (triangular_mf.widths.detach().numpy() == widths).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_sigmas_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        widths = np.array([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        triangular_mf = Triangular(in_features=elements.shape[1], widths=widths, sort_by=None)
        # we will now update the widths to be abs. value
        widths = np.abs(widths)
        centers = triangular_mf.centers.detach().numpy()
        mu_pytorch = triangular_mf(torch.tensor(elements))
        mu_numpy = TriangularNumpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterwards
        assert (triangular_mf.centers.detach().numpy() == centers).all()
        assert np.isclose(triangular_mf.widths.detach().numpy(), widths).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()

    def test_multi_input_with_both_given(self):
        elements = torch.tensor([[0.41737163], [0.78705574], [0.40919196], [0.72005216]])
        centers = np.array([-0.5, -0.25, 0.25, 0.5, 0.75])
        widths = np.array([-0.1, 0.25, -0.5, 0.75, 1.0])  # any < 0 sigma values will be > 0 sigma values
        triangular_mf = Triangular(in_features=elements.shape[1], centers=centers, widths=widths)
        # we will now update the widths to be abs. value
        widths = np.abs(widths)
        mu_pytorch = triangular_mf(torch.tensor(elements))
        mu_numpy = TriangularNumpy(elements.detach().numpy(), centers, widths)

        # make sure the Gaussian parameters are still identical afterwards
        assert (triangular_mf.centers.detach().numpy() == centers).all()
        assert np.isclose(triangular_mf.widths.detach().numpy(), widths).all()
        # the outputs of the PyTorch and Numpy versions should be approx. equal
        assert np.isclose(mu_pytorch.squeeze(dim=1).detach().numpy(), mu_numpy, rtol=1e-6).all()
