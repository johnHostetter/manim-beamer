import unittest
import numpy as np

# actual implementation of CLIP that may break and is written in PyTorch
from soft.fuzzy.online.unsupervised.cluster.ecm import ECM as newECM, general_euclidean_distance as newDistance
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
        assert np.isclose(oldDistance(x, y), newDistance(x, y))

    def test_clip_output(self):
        """
        The CLIP that is originally defined without using PyTorch should match the newly defined
        output that uses PyTorch to produce the results (i.e., the new CLIP directly induces
        Gaussian PyTorch modules).

        Uses sample input from the Cart Pole FCQL demo.

        Returns:

        """
        Dthr = 0.4
        train_X = np.load('ecm_input.npy')
        old_clusters = oldECM(train_X, [], Dthr)
        old_centers = [cluster.center for cluster in old_clusters]
        old_widths = [cluster.radius for cluster in old_clusters]

        new_clusters = newECM(train_X, [], Dthr)
        # new_centers = [cluster.center for cluster in new_clusters]
        # new_widths = [cluster.radius for cluster in new_clusters]

        assert len(old_centers) == len(new_clusters.centers.detach().numpy())  # results should be of same size
        assert np.isclose(old_centers, new_clusters.centers.detach().numpy()).all()  # approx equal centers
        assert len(old_widths) == len(new_clusters.widths.detach().numpy())  # results should be of same size
        assert np.isclose(old_widths, new_clusters.widths.detach().numpy()).all()  # approx equal sigmas