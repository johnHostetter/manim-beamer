import os
import torch
import pathlib
import unittest
import numpy as np

# actual implementation of CLIP that may break and is written in PyTorch
from soft.fuzzy.online.unsupervised.cluster.ecm import apply_evolving_clustering_method as newECM, general_euclidean_distance as newDistance
# local implementations of CLIP that we know work but are written in Numpy
from tests.fuzzy.online.unsupervised.cluster.ecm import ECM as oldECM, general_euclidean_distance as oldDistance


class TestECM(unittest.TestCase):
    def test_general_euclidean_distance(self):
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:

        """
        x = np.array([0.5, 0.8])
        y = np.array([0.9, 2.5])
        assert np.isclose(oldDistance(x, y), newDistance(torch.tensor(np.array([x])), torch.tensor(y)))

    def test_ecm_output(self):
        """
        The ECM that is originally defined without using PyTorch should identify the same
        number of exemplars as the PyTorch implementation (i.e., the new ECM directly induces
        Fuzzy Set Base PyTorch modules). However, the identified CENTERS and WIDTHS should
        be UNEQUAL, since the original implementation had some minor mistakes.

        Uses sample input from the Cart Pole FCQL demo.

        Returns:

        """
        Dthr = 0.4
        directory = pathlib.Path(__file__).parent.resolve()
        file_location = os.path.join(directory, 'ecm_input.npy')
        train_X = np.load(file_location)
        old_clusters = oldECM(train_X, [], Dthr)
        old_centers = np.array([cluster.center.tolist() for cluster in old_clusters])
        old_widths = [cluster.radius for cluster in old_clusters]

        new_clusters = newECM(torch.tensor(train_X), config={'dthr': Dthr})

        assert len(old_centers) == len(new_clusters.centers.detach().numpy())  # results should be of same size
        assert not np.isclose(old_centers, new_clusters.centers.detach().numpy()).all()  # approx unequal centers
        assert len(old_widths) == len(new_clusters.widths.detach().numpy())  # results should be of same size
        assert not np.isclose(old_widths, new_clusters.widths.detach().numpy()).all()  # approx equal sigmas
