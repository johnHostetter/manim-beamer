import unittest
import numpy as np

# actual implementation of CLIP that may break and is written in PyTorch
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP as newCLIP, regulator as newR
# local implementations of CLIP that we know work but are written in Numpy
from tests.fuzzy.online.unsupervised.granulation.clip import CLIP as oldCLIP, R as oldR


class TestCLIP(unittest.TestCase):
    def test_regulator(self):
        """
        The regulator function implemented using PyTorch should perform
        identical functionality to the one implemented in Numpy.

        Returns:

        """
        assert np.isclose(oldR(1.0, 1.0), newR(1.0, 1.0))

    def test_clip_output(self):
        """
        The CLIP that is originally defined without using PyTorch should match the newly defined
        output that uses PyTorch to produce the results (i.e., the new CLIP directly induces
        Gaussian PyTorch modules).

        Uses sample input from the Cart Pole FCQL demo.

        Returns:

        """

        train_X = np.load('clip_input.npy')
        train_X_mins = train_X.min(axis=0)
        train_X_maxes = train_X.max(axis=0)

        oldCLIP_terms = oldCLIP(train_X, train_X_mins, train_X_maxes)
        newCLIP_terms = newCLIP(train_X, train_X_mins, train_X_maxes)

        for linguistic_variable_idx, linguistic_variable in enumerate(zip(oldCLIP_terms, newCLIP_terms)):
            original_results, new_results = linguistic_variable[0], linguistic_variable[1]
            original_centers = [term['center'] for term in original_results]
            original_sigmas = [term['sigma'] for term in original_results]
            assert len(original_centers) == len(new_results.centers.detach().numpy())  # results should be of same size
            assert np.isclose(original_centers, new_results.centers.detach().numpy()).all()  # approx equal centers
            assert len(original_sigmas) == len(new_results.widths.detach().numpy())  # results should be of same size
            assert np.isclose(original_sigmas, new_results.widths.detach().numpy()).all()  # approx equal sigmas
