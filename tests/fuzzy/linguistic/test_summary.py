import torch
import unittest
import numpy as np

from soft.fuzzy.sets import Gaussian
from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA
from soft.fuzzy.linguistic.summary import Summary, Query, most_quantifier as Q


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

    def test_owa_with_importance(self):
        importance = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert importance.sum() == 1.0
        in_features = len(importance)
        p = in_features
        x = torch.tensor([0, 0.7, 1.0, 0.2])
        sorted_x = torch.sort(x, descending=True)  # namedtuple with 'values' and 'indices' properties
        assert torch.isclose(sorted_x.values, torch.tensor([1.0000, 0.7000, 0.2000, 0.0000])).all()
        sorted_importance = importance[sorted_x.indices]
        assert torch.isclose(sorted_importance, torch.tensor([0.1000, 0.3000, 0.4000, 0.2000])).all()

        denominator = sorted_importance.sum()
        weights = []
        for j in range(p):
            left_side = Q(sorted_importance[:j + 1].sum() / denominator)
            right_side = Q(sorted_importance[:j].sum() / denominator)
            weights.append((left_side - right_side).item())
        weights = torch.tensor(weights)

        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert torch.isclose(owa(x), torch.tensor(0.30))

    def test_summarizer_membership(self):
        """
        The membership of the summarizer should be equal to the minimum membership found across the list of fuzzy sets
        seen in the summarizer argument.

        Returns:
            None
        """
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        assert torch.isclose(summary.summarizer_membership(x), torch.tensor(0.5272924900054932))

    def test_summarizer_membership_query(self):
        """
        The membership of the summarizer should be equal to the minimum membership found across the list of fuzzy sets
        seen in the summarizer argument.

        Returns:
            None
        """
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want to constrain that the second attribute has to satisfy the following
        query = Query(Gaussian(1, centers=0.3, sigmas=0.3), 1)
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.5272924900054932))  # it should
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))

    def test_degree_of_truth(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))
        X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
        assert torch.isclose(summary.r(X, query=query), torch.tensor(0.5483109354972839))
        assert torch.isclose(summary.degree_of_truth(X, query=query), torch.tensor(0.49662184715270996))

    def test_degree_of_imprecision(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))
        X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
        assert torch.isclose(summary.degree_of_imprecision(X, alpha=0.3, query=query), torch.tensor(1 / 4))

    def test_degree_of_covering(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))
        X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
        assert torch.isclose(summary.degree_of_covering(X, alpha=0.3, query=query), torch.tensor(2 / 3))

    def test_degree_of_appropriateness(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))
        X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
        assert torch.isclose(summary.degree_of_appropriateness(X, alpha=0.3, query=query),
                             torch.tensor(0.006944441236555576))

    def test_length(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        assert torch.isclose(summary.length(), torch.tensor(1/2))

    def test_degree_of_validity(self):
        summarizer = [Gaussian(1, centers=0.8, sigmas=0.25), Gaussian(1, centers=0.4, sigmas=0.25)]
        summary = Summary(summarizer, Q, None)
        x = torch.tensor([1., 0.5])
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, sigmas=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))
        X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
        assert torch.isclose(summary.degree_of_validity(X, alpha=0.3, query=query),
                             torch.tensor(0.3840465843677521))
