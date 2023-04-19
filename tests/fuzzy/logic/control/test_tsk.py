"""
Test the ZeroOrderTSK is working as intended, such as its output is correctly calculated.
"""
import unittest

import torch

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.logic.control.tsk import ZeroOrderTSK
from soft.fuzzy.relation.tnorm import AlgebraicProduct

set_rng(0)


class TestTSK(unittest.TestCase):
    """
    Test the zero-order TSK neuro-fuzzy network.
    """

    def test_gradient_1(self):
        """
        First test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()
        # first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(torch.tensor([[0, 1], [1, 2], [2, 3]]).float())
        actual_result = input_data.unsqueeze(dim=-1) - centers.T
        expected_result = torch.tensor([[[1.2000, 0.2000, -0.8000],
                                         [-0.8000, -1.8000, -2.8000]],
                                        [[1.1000, 0.1000, -0.9000],
                                         [-0.7000, -1.7000, -2.7000]],
                                        [[2.1000, 1.1000, 0.1000],
                                         [-0.9000, -1.9000, -2.9000]],
                                        [[2.7000, 1.7000, 0.7000],
                                         [-0.8500, -1.8500, -2.8500]],
                                        [[1.7000, 0.7000, -0.3000],
                                         [-0.7500, -1.7500, -2.7500]]])

        assert torch.isclose(actual_result, expected_result).all()

    def test_gradient_2(self):
        """
        Second test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        value_1 = torch.nn.Parameter(torch.tensor([0, 1]).float())
        value_3 = 2 ** value_1
        assert value_3.grad_fn is not None

    def test_tsk(self):
        """
        Test the zero-order TSK neuro-fuzzy network.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()
        actual_y = torch.tensor([1.5, 0.6, 0.9, 0.7, 1.3]).float()

        antecedents = [
            Gaussian(4, centers=torch.tensor([1.2, 3.0, 5.0, 7.0]).float(),
                     widths=torch.tensor([0.1, 0.4, 0.6, 0.8]).float()),
            Gaussian(4, centers=torch.tensor([0.2, 0.6, 0.9, 1.2]).float(),
                     widths=torch.tensor([0.4, 0.4, 0.5, 0.45]).float())
        ]

        # check that antecedents were correctly created
        assert (antecedents[0].centers == torch.tensor([1.2, 3.0, 5.0, 7.0])).all()
        assert (antecedents[0].widths == torch.tensor([0.1, 0.4, 0.6, 0.8])).all()
        assert (antecedents[1].centers == torch.tensor([0.2, 0.6, 0.9, 1.2])).all()
        assert (antecedents[1].widths == torch.tensor([0.4, 0.4, 0.5, 0.45])).all()

        rules = {
            frozenset({(0, 0), (1, 0)}), frozenset({(0, 1), (1, 0)}),
            frozenset({(0, 1), (1, 1)}), frozenset({(1, 1), (1, 1)})
        }
        knowledge_base = expert_design(antecedents, rules, config={})

        rule_vertex = knowledge_base.graph.vs.find(relation_eq=AlgebraicProduct)
        assert rule_vertex['relation'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'id' in rule_vertex.attributes()  # it has a unique id

        rule_vertices = knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)
        assert len(rule_vertices) == len(rules)  # number of rule vertices should equal len(rules)

        # there should be 2 rules that use (1, 1);
        # the last rule has been simplified (redundant mention of condition)
        assert knowledge_base[(1, 1)] == {
            AlgebraicProduct: [frozenset({(0, 1), (1, 1)}), frozenset({(1, 1)})]
        }

        # the rules we have added should exist how we expected them
        assert knowledge_base.edges(AlgebraicProduct) == rules

        knowledge_base.attributes(rule_vertex['name'])
        flc = ZeroOrderTSK(out_features=actual_y.ndim, knowledge_base=knowledge_base,
                           input_trainable=True)
        predicted_y = flc(input_data)
        assert (predicted_y == torch.zeros(input_data.shape[0])).all()

        assert (flc.input_granulation.centers == torch.tensor([
            [1.2000, 3.0000, 5.0000, 7.0000],
            [0.2000, 0.6000, 0.9000, 1.2000]
        ])).all()
        assert (flc.input_granulation.sigmas == torch.tensor([
            [0.1000, 0.4000, 0.6000, 0.8000],
            [0.4000, 0.4000, 0.5000, 0.4500]
        ])).all()
        assert (flc.consequences() == torch.zeros((len(rules), flc.out_features()))).all()
