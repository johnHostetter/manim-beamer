"""
Test various components necessary in implementing the
Categorical Learning Induced Partitioning algorithm.
"""
import os
import pathlib
import unittest

import torch
import numpy as np

from utils.reproducibility import set_rng, default_configuration
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.online.unsupervised.granulation.clip import (
    find_indices_to_closest_neighbors,
)
from soft.fuzzy.online.unsupervised.granulation.clip import (
    regulator,
    apply_categorical_learning_induced_partitioning as CLIP,
)


class TestCLIP(unittest.TestCase):
    """
    Test the Categorical Learning Induced Partitioning algorithm, and its supporting functions
    such as the 'regular' function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()

    def test_regulator(self):
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:
            None
        """
        assert regulator(sigma_1=1.0, sigma_2=1.0) == 1.0
        assert regulator(sigma_1=0.5, sigma_2=1.0) == 0.75
        assert regulator(sigma_1=1.0, sigma_2=0.5) == 0.75
        assert regulator(sigma_1=0.5, sigma_2=0.5) == 0.5
        assert regulator(sigma_1=0, sigma_2=1.0) == 0.5

    def test_find_indices_to_closest_neighbors(self):
        """
        Test that the 'find_indices_to_closest_neighbors' function correctly identifies the
        data observation's left and right neighbor indices.

        Returns:
            None
        """
        # a new cluster is created in the input dimension based on the presented value
        dimension = 1
        element = torch.tensor([0.0, 3.0, 2.0, 3])
        terms = [[], Gaussian(in_features=4, centers=torch.tensor([-2.0, 2.0, 4.0, 6]))]
        left_neighbor_idx, right_neighbor_idx = find_indices_to_closest_neighbors(
            element, terms, dimension
        )

        assert left_neighbor_idx == 1
        assert right_neighbor_idx == 2

    def test_clip_on_random_data(self):
        """
        Testing how CLIP performs when given some random data.

        Returns:
            None
        """
        set_rng(0)
        directory = pathlib.Path(__file__).parent.resolve()
        file_path = os.path.join(directory, "random_train_data.npy")
        input_data = np.load(file_path)
        linguistic_terms = CLIP(torch.tensor(input_data), config=self.config)
        expected_terms = [
            Gaussian(
                in_features=3,
                centers=[0.5488135, 0.96366276, 0.0202184],
                widths=[0.38547637, 0.3542887, 0.38547637],
            ),
            Gaussian(
                in_features=4,
                centers=[0.71518937, 0.38344152, 0.10204481, 0.95274901],
                widths=[0.25629429, 0.27357153, 0.27357153, 0.25629429],
            ),
            Gaussian(
                in_features=3,
                centers=[0.60276338, 0.07103606, 0.94374808],
                widths=[0.33559839, 0.40241628, 0.33559839],
            ),
            Gaussian(
                in_features=3,
                centers=[0.54488318, 0.891773, 0.0871293],
                widths=[0.34311335, 0.32540311, 0.34311335],
            ),
        ]
        for linguistic_term, expected_term in zip(linguistic_terms, expected_terms):
            assert linguistic_term.in_features, expected_term.in_features
            assert torch.isclose(
                linguistic_term.centers.float(), expected_term.centers.float()
            ).all()
            assert torch.isclose(
                linguistic_term.widths.float(), expected_term.widths.float()
            ).all()

        # compare_results(oldCLIP_terms, newCLIP_terms)
