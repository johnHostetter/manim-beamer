import torch
import unittest

from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA


class TestOrderedWeightedAggregation(unittest.TestCase):
    def test_weight_vector_sums_to_one(self):
        """
        An OWA operator module should be created when given a weight vector that sums to 1.

        Returns:
            None
        """
        weight = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()

    def test_weight_vector_not_sum_to_one(self):
        """
        Attempting to create a OWA with a weight vector that does not sum to 1 should throw an AttributeError exception.

        Returns:
            None
        """
        weight = torch.tensor([0.2, 0.3, 0.1, 0.3])
        assert torch.isclose(weight.sum(), torch.tensor(0.9))
        try:
            in_features = len(weight)
            OWA(in_features, weight)
            assert False
        except AttributeError:
            assert True  # an AttributeError exception should be thrown when the weight does not sum to 1

    def test_owa_calculation_1(self):
        """
        A OWA operator should sort the argument vector to produce an 'ordered argument vector', then
        calculate the dot product between the weight vector and ordered argument vector.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weight = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        argument_vector = torch.tensor([0.6, 1.0, 0.3, 0.5])
        assert torch.isclose(owa(argument_vector), torch.tensor(0.55))

    def test_owa_calculation_2(self):
        """
        A OWA operator should sort the argument vector to produce an 'ordered argument vector', then
        calculate the dot product between the weight vector and ordered argument vector.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weight = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        argument_vector = torch.tensor([0, 0.7, 1.0, 0.2])
        assert torch.isclose(owa(argument_vector), torch.tensor(0.43))
