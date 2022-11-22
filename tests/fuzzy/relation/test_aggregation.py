import torch
import unittest

from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA


class TestOrderedWeightedAggregation(unittest.TestCase):
    def test_in_features_not_equal_to_weight_vector(self):
        weight = torch.tensor([0.2, 0.3, 0.1])
        assert torch.isclose(weight.sum(), torch.tensor(0.6))
        try:
            in_features = 4
            OWA(in_features, weight)
            assert False
        except AttributeError:
            assert True  # an AttributeError exception should be thrown when the weight vector != in_features

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

    def test_orness_1(self):
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 1 means the OWA operator is the 'or' operator, and this occurs when the first
        element of the weight vector is equal to 1 and all other elements in the weight vector are zero.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weight = torch.tensor([1.0, 0., 0., 0.])
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        assert owa.orness() == 1.0

    def test_orness_2(self):
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 0 means the OWA operator is the 'and' operator, and this occurs when the last
        element of the weight vector is equal to 1 and all other elements in the weight vector are zero.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weight = torch.tensor([0., 0., 0., 1.])
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        assert owa.orness() == 0.0

    def test_orness_3(self):
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 1 means the OWA operator is the 'or' operator, and this occurs when the first
        element of the weight vector is equal to 1 and all other elements in the weight vector are zero.

        This test follows an example from the original paper.

        Returns:
            None
        """
        n = 4
        weight = torch.tensor([1 / n] * n)
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        assert owa.orness() == 0.5  # will be 0.5 for any number of 'n' if the weight vector consists of 1/n values

    def test_orness_4(self):
        """
        The 'orness' measure can be misleading when Ordered Weighted Averaging operators are defined with
        certain weight vectors as illustrated here. Both weight vectors should have a 'orness' of 0.5,
        despite being clearly different. The measure of dispersion is introduced to address this.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weight = torch.tensor([0., 0., 1., 0., 0.])  # considered to be more volatile and uses less input
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa1 = OWA(in_features, weight)
        assert torch.isclose(owa1.weight, weight).all()
        n = 5
        weight = torch.tensor([1 / n] * n)
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa2 = OWA(in_features, weight)
        assert torch.isclose(owa2.weight, weight).all()
        assert owa1.orness() == 0.5
        assert owa2.orness() == 0.5  # will be 0.5 for any number of 'n' if the weight vector consists of 1/n values
        assert owa1.orness() == owa2.orness()

    def test_dispersion_1(self):
        """
        This scenario represents the minimum dispersion.

        Returns:
            None
        """
        weight = torch.tensor([0., 0., 1., 0., 0.])  # considered to be more volatile and uses less input
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        assert owa.dispersion() == 0.

    def test_dispersion_2(self):
        """
        This scenario represents the maximum dispersion and occurs when the entries in the weight vector
        are 1/n where n is the number of elements in the weight vector.

        Returns:
            None
        """
        n = 5
        weight = torch.tensor([1 / n] * n)
        assert weight.sum() == 1.0
        in_features = len(weight)
        owa = OWA(in_features, weight)
        assert torch.isclose(owa.weight, weight).all()
        assert owa.dispersion() == torch.log(torch.tensor(n))
