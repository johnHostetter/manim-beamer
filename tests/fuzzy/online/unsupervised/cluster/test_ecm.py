"""
Test the Evolving Clustering Method and its accompanying functions.
"""
import os
import pathlib
import unittest

import torch
import numpy as np

from utils.reproducibility import default_configuration
from soft.fuzzy.online.unsupervised.cluster.ecm import (
    apply_evolving_clustering_method as ECM,
    general_euclidean_distance,
)
from soft.fuzzy.sets.continuous import ContinuousFuzzySet


class TestECM(unittest.TestCase):
    """
    Test the Evolving Clustering Method and its accompanying functions, such as the general
    Euclidean distance metric, and the algorithm itself.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()

    def test_general_euclidean_distance(self):
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:

        """
        vector_1 = np.array([0.5, 0.8])
        vector_2 = np.array([0.9, 2.5])
        distance = general_euclidean_distance(
            torch.tensor(np.array([vector_1])), torch.tensor(vector_2)
        ).float()
        expected_distance = torch.tensor([1.2349089]).float()
        assert torch.isclose(distance, expected_distance).all()

    def test_ecm_output(self):
        """
        The ECM that is originally defined without using PyTorch should identify the same
        number of exemplars as the PyTorch implementation (i.e., the new ECM directly induces
        Fuzzy Set Base PyTorch modules). However, the identified CENTERS and WIDTHS should
        be UNEQUAL, since the original implementation had some minor mistakes.

        Uses sample input from the Cart Pole FCQL demo.

        Returns:

        """
        directory = pathlib.Path(__file__).parent.resolve()
        file_location = os.path.join(directory, "ecm_input.npy")
        input_data = np.load(file_location)

        clusters = ECM(torch.tensor(input_data), config=self.config)
        expected_clusters_centers = torch.tensor(
            [
                [0.05652926, 0.06268818, -0.11167774, -0.38759786],
                [-0.20636721, -1.3460023, -0.15555668, 0.53961074],
                [-0.01227049, 0.96697235, -0.08816208, -1.5092062],
                [-1.5831373, -1.1857209, 0.133384, -0.00864309],
                [0.6316188, 1.5248702, 0.14401233, -0.01392129],
                [1.8898193, 2.4484441, 0.16448256, 0.00566044],
                [-1.0552928, -2.263982, 0.02294253, 1.1614264],
                [-1.3587813, 0.37505904, 0.17042659, -0.8567319],
                [0.05848482, -0.9122014, 0.15639654, 1.8573731],
                [-2.1443288, -2.7417984, -0.09376517, 0.31661162],
                [0.8912618, 0.48925343, 0.14131558, 1.0544153],
            ]
        )
        expected_clusters_widths = torch.tensor(
            [
                0.6810045,
                0.6918813,
                0.6853437,
                0.6883185,
                0.6896569,
                0.36993876,
                0.68684465,
                0.67583203,
                0.563162,
                0.0,
                0.0,
            ]
        )
        expected_clusters = ContinuousFuzzySet(
            in_features=clusters.in_features,
            centers=expected_clusters_centers,
            widths=expected_clusters_widths,
        )
        assert torch.isclose(clusters.centers, expected_clusters_centers).all()
        assert torch.isclose(clusters.widths, expected_clusters_widths).all()
        # test that ContinuousFuzzySet was properly created
        assert clusters.in_features == expected_clusters.in_features
        assert torch.isclose(clusters.centers, expected_clusters.centers).all()
        assert torch.isclose(clusters.widths, expected_clusters.widths).all()
