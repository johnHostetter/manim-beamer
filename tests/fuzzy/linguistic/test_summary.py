import torch
import unittest
import numpy as np

from soft.fuzzy.sets import Gaussian
from soft.fuzzy.linguistic.summary import most_quantifier as Q


class TestSummary(unittest.TestCase):
    def test_most_quantifier(self):
        assert Q(1.0) == 1.0
        assert Q(0.8) == 1.0
        assert np.isclose(Q(0.7), 0.8)
        assert Q(0.3) == 0.0

    def test_linguistic_quantified_proposition(self):
        with torch.no_grad():
            elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
            n_inputs = 1
            property_mf = Gaussian(n_inputs, centers=[0.8], sigmas=[0.4])
            assert property_mf.centers.detach().numpy() == 0.8
            assert property_mf.sigmas.detach().numpy() == 0.4
            mu = property_mf(elements)
            x = mu.sum() / elements.nelement()
            assert torch.isclose(x, torch.tensor(0.7572454810142517))  # compare to ground truth value
            truth_of_proposition = Q(x)
            assert torch.isclose(truth_of_proposition, torch.tensor(0.9145))  # compare to ground truth value

    def test_linguistic_quantified_proposition_with_importance(self):
        with torch.no_grad():
            elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
            n_inputs = 1
            property_mf = Gaussian(n_inputs, centers=[0.8], sigmas=[0.4])
            importance_mf = Gaussian(n_inputs, centers=[0.6], sigmas=[0.2])
            assert property_mf.centers.detach().numpy() == 0.8
            assert property_mf.sigmas.detach().numpy() == 0.4
            assert importance_mf.centers.detach().numpy() == 0.6
            assert importance_mf.sigmas.detach().numpy() == 0.2
            property_mu = property_mf(elements)
            importance_mu = importance_mf(elements)
            t_norm_results = property_mu * importance_mu
            assert torch.isclose(t_norm_results,
                                 torch.tensor([0.7316157, 0.77880085, 0.3678795, 0.09901349, 0.5989963,
                                               0.26497352, 0.8187308, 0.00193045])).all()
            assert torch.isclose(importance_mu.sum(), torch.tensor(4.4135942459106445))
            x = t_norm_results.sum() / importance_mu.sum()
            assert torch.isclose(x, torch.tensor(0.8296958208084106))  # compare to ground truth value
            truth_of_proposition = Q(x)
            assert torch.isclose(truth_of_proposition, torch.tensor(1.0))  # compare to ground truth value
