"""
Test the aggregation operator called the ordered weighted averaging operator.
"""
import unittest

import torch

from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA


class TestOrderedWeightedAggregation(unittest.TestCase):
    """
    Unit tests that check the ordered weighted averaging operator is working as intended.
    """

    def test_in_features_not_equal_to_weight_vector(self) -> None:
        """
        Test that when trying to create a OrderedWeightedAveraging object with a weights vector
        that does not equal the number of input features specified, that an AttributeError is
        thrown.

        Returns:
            None
        """
        weights = torch.tensor([0.2, 0.3, 0.1])
        assert torch.isclose(weights.sum(), torch.tensor(0.6))
        try:
            in_features = 4
            OWA(in_features, weights)
            assert False
        except AttributeError:
            # an AttributeError exception should be thrown when the weights vector != in_features
            assert True

    def test_weight_vector_sums_to_one(self) -> None:
        """
        An OWA operator module should be created when given a weights vector that sums to 1.

        Returns:
            None
        """
        weights = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()

    def test_weight_vector_not_sum_to_one(self) -> None:
        """
        Attempting to create a OWA with a weight vector that does
        not sum to 1 should throw an AttributeError exception.

        Returns:
            None
        """
        weights = torch.tensor([0.2, 0.3, 0.1, 0.3])
        assert torch.isclose(weights.sum(), torch.tensor(0.9))
        try:
            in_features = len(weights)
            OWA(in_features, weights)
            assert False
        except AttributeError:
            # an AttributeError exception should be thrown when the weights do not sum to 1
            assert True

    def test_owa_calculation_1(self) -> None:
        """
        A OWA operator should sort the argument vector to produce an 'ordered argument vector',
        then calculate the dot product between the weights vector and ordered argument vector.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weights = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        argument_vector = torch.tensor([0.6, 1.0, 0.3, 0.5])
        assert torch.isclose(owa(argument_vector), torch.tensor(0.55))

    def test_owa_calculation_2(self) -> None:
        """
        A OWA operator should sort the argument vector to produce an 'ordered argument vector',
        then calculate the dot product between the weights vector and ordered argument vector.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weights = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        argument_vector = torch.tensor([0, 0.7, 1.0, 0.2])
        assert torch.isclose(owa(argument_vector), torch.tensor(0.43))

    def test_orness_1(self) -> None:
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 1 means the OWA operator is the 'or' operator, and this occurs when the first
        element of the weights vector is equal to 1 and all other elements in the weights
        vector are zero.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weights = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert owa.orness() == 1.0

    def test_orness_2(self) -> None:
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 0 means the OWA operator is the 'and' operator, and this occurs when the last
        element of the weights vector is equal to 1 and all other elements in the weights
        vector are zero.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        weights = torch.tensor([0.0, 0.0, 0.0, 1.0])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert owa.orness() == 0.0

    def test_orness_3(self) -> None:
        """
        The degree to which the Ordered Weighted Averaging (OWA) operator is the 'or' operator.

        A degree of 1 means the OWA operator is the 'or' operator, and this occurs when the first
        element of the weights vector is equal to 1 and all other elements in the weights
        vector are zero.

        This test follows an example from the original paper.

        Returns:
            None
        """
        number_of_elements = 4
        weights = torch.tensor([1 / number_of_elements] * number_of_elements)
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        # will be 0.5 for any number of 'number_of_elements' if the weights vector
        # consists of 1/number_of_elements values
        assert owa.orness() == 0.5

    def test_orness_4(self) -> None:
        """
        The 'orness' measure can be misleading when Ordered Weighted Averaging operators are
        defined with certain weights vectors as illustrated here. Both weights vectors should
        have a 'orness' of 0.5, despite being clearly different. The measure of dispersion is
        introduced to address this.

        This test replicates an example from the original paper.

        Returns:
            None
        """
        # considered to be more volatile and uses less input
        weights = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa1 = OWA(in_features, weights)
        assert torch.isclose(owa1.weights, weights).all()
        number_of_elements = 5
        weights = torch.tensor([1 / number_of_elements] * number_of_elements)
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa2 = OWA(in_features, weights)
        assert torch.isclose(owa2.weights, weights).all()
        # will be 0.5 for any number of 'number_of_elements' if the weights vector
        # consists of 1/number_of_elements values
        assert owa1.orness() == 0.5
        assert owa2.orness() == 0.5
        assert owa1.orness() == owa2.orness()

    def test_dispersion_1(self) -> None:
        """
        This scenario represents the minimum dispersion.

        Returns:
            None
        """
        # considered to be more volatile and uses less input
        weights = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert owa.dispersion() == 0.0

    def test_dispersion_2(self) -> None:
        """
        This scenario represents the maximum dispersion and occurs when the entries in the
        weights vector are 1/number_of_elements where number_of_elements is the number of
        elements in the weights vector.

        Returns:
            None
        """
        number_of_elements = 5
        weights = torch.tensor([1 / number_of_elements] * number_of_elements)
        assert weights.sum() == 1.0
        in_features = len(weights)
        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert owa.dispersion() == torch.log(torch.tensor(number_of_elements))
