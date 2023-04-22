"""
Test the Mamdani FLC is working as intended, such as its output is correctly calculated.
"""
import unittest

import torch

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.logic.control.controller import Mamdani

set_rng(0)


class TestMamdani(unittest.TestCase):
    """
    Test the Mamdani neuro-fuzzy network.
    """

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

        consequents = [
            Gaussian(
                in_features=4,
                centers=torch.tensor([1.2, 3.0, 5.0, 7.0]).float(),
                widths=torch.tensor([0.1, 0.4, 0.6, 0.8]).float()
            ),
            Gaussian(
                in_features=3,
                centers=torch.tensor([0.2, 0.6, 0.9]).float(),
                widths=torch.tensor([0.4, 0.4, 0.5]).float()
            )
        ]
        rules = {
            Rule(premise=frozenset({(0, 0), (1, 0)}), consequence=frozenset({(2, 0), (3, 1)}),
                 implication=AlgebraicProduct),
            Rule(premise=frozenset({(0, 1), (1, 0)}), consequence=frozenset({(2, 1), (3, 2)}),
                 implication=AlgebraicProduct),
            Rule(premise=frozenset({(0, 1), (1, 1)}), consequence=frozenset({(2, 0), (3, 0)}),
                 implication=AlgebraicProduct),
        }

        knowledge_base = expert_design(antecedents, consequents, rules=rules, config={})

        rule_vertex = knowledge_base.graph.vs.find(type_eq=AlgebraicProduct)
        assert rule_vertex['type'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'type' in rule_vertex.attributes()  # it has 'type' attribute

        rule_vertices = knowledge_base.graph.vs.select(type_eq=AlgebraicProduct)
        assert len(rule_vertices) == len(rules)  # number of rule vertices should equal len(rules)

        # the rules we have added should exist how we expected them
        assert knowledge_base.get_fuzzy_logic_rules() == rules

        flc = Mamdani(knowledge_base=knowledge_base, learning_rate=1e-3)
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
