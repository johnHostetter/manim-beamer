import os
import torch
import pathlib
import unittest
import numpy as np

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian

# local implementations of CLIP that we know work but are written in Numpy
from tests.fuzzy.online.unsupervised.granulation.clip import CLIP as oldCLIP, R as oldR
from soft.fuzzy.online.unsupervised.granulation.clip import find_indices_to_closest_neighbors
# actual implementation of CLIP that may break and is written in PyTorch
from soft.fuzzy.online.unsupervised.granulation.clip import apply_categorical_learning_induced_partitioning as newCLIP, regulator as newR

set_rng(0)


def compare_results(oldCLIP_terms, newCLIP_terms):
    """
    Compare the results between the original CLIP implementation and the new
    CLIP implementation (PyTorch).

    Args:
        oldCLIP_terms: The terms that are produced by the original CLIP implementation.
        newCLIP_terms: The terms that are produced by the new CLIP implementation (PyTorch).
        eq: Whether the results should be equal, by default is True. Some mistakes in the
            original implementation however were noticed afterwards, and so in some settings,
            this should be set to be False.

    Returns:
        None
    """
    for linguistic_variable_idx, linguistic_variable in enumerate(zip(oldCLIP_terms, newCLIP_terms)):
        original_results, new_results = linguistic_variable[0], linguistic_variable[1]
        original_centers = [term['center'] for term in original_results]
        original_sigmas = [term['sigma'] for term in original_results]

        assert len(original_centers) == len(new_results.centers.detach().numpy())  # results should be of same size
        assert np.isclose(original_centers, new_results.centers.detach().numpy()).all()  # approx equal centers
        assert len(original_sigmas) == len(new_results.widths.detach().numpy())
        assert np.isclose(original_sigmas, new_results.widths.detach().numpy()).all()


class TestCLIP(unittest.TestCase):

    def test_regulator(self):
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:
            None
        """
        assert np.isclose(oldR(1.0, 1.0), newR(1.0, 1.0))

    def test_find_indices_to_closest_neighbors(self):
        """
        Test that the 'find_indices_to_closest_neighbors' function correctly identifies the
        data observation's left and right neighbor indices.

        Returns:
            None
        """
        # a new cluster is created in the input dimension based on the presented value
        dimension = 1
        x = torch.tensor([0., 3., 2., 3])
        terms = [[], Gaussian(in_features=4, centers=torch.tensor([-2., 2., 4., 6]))]
        left_neighbor_idx, right_neighbor_idx = find_indices_to_closest_neighbors(x, terms, dimension)

        assert left_neighbor_idx == 1
        assert right_neighbor_idx == 2

    def test_clip_on_random_data(self):
        """
        Testing how CLIP performs when given some random data.

        Returns:
            None
        """
        directory = pathlib.Path(__file__).parent.resolve()
        file_path = os.path.join(directory, 'random_train_data.npy')
        train_X = np.load(file_path)
        oldCLIP_terms = oldCLIP(train_X, train_X.min(axis=0), train_X.max(axis=0))
        newCLIP_terms = newCLIP(torch.tensor(train_X), config={'eps': 0.2, 'kappa': 0.6})

        compare_results(oldCLIP_terms, newCLIP_terms)
