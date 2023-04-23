"""
Test the Mamdani FLC is working as intended, such as its output is correctly calculated.
"""
import unittest

import torch
import numpy as np

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.logic.control.controller import Mamdani
from soft.fuzzy.relation.tnorm import AlgebraicProduct

set_rng(0)


class TestMamdani(unittest.TestCase):
    """
    Test the Mamdani neuro-fuzzy network.
    """

    def test_mamdani(self):
        """
        Test the Mamdani neuro-fuzzy network.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()

        antecedents = [
            Gaussian(in_features=4, centers=torch.tensor([1.2, 3.0, 5.0, 7.0]).float(),
                     widths=torch.tensor([0.1, 0.4, 0.6, 0.8]).float()),
            Gaussian(in_features=4, centers=torch.tensor([0.2, 0.6, 0.9, 1.2]).float(),
                     widths=torch.tensor([0.4, 0.4, 0.5, 0.45]).float())
        ]

        consequents = [
            Gaussian(
                in_features=4,
                centers=torch.tensor([0.5, 0.3, 0.1, 0.9]).float(),
                widths=torch.tensor([0.1, 0.4, 0.6, 0.8]).float()
            ),
            Gaussian(
                in_features=3,
                centers=torch.tensor([-0.2, -0.7, -0.9]).float(),
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

        # check the intra-dimensionality of the input & output spaces are correctly calculated
        assert all(knowledge_base.intra_dimensions(True) == np.array([
            term.in_features for term in antecedents]))
        assert all(knowledge_base.intra_dimensions(False) == np.array([
            term.in_features for term in consequents]))
        # the above is required to generate the correct links shape for fuzzy inference

        # check the variable dimensionality of the input & output spaces is correctly calculated
        assert knowledge_base.variable_dimensions(True) == len(antecedents)
        assert knowledge_base.variable_dimensions(False) == len(consequents)
        # the above is required to generate the correct links shape for fuzzy inference

        flc = Mamdani(
            out_features=len(consequents), knowledge_base=knowledge_base, learning_rate=1e-3)

        # check that the intermediate structure of the Mamdani FLC is correctly built
        input_links, input_offset = knowledge_base.matrix(AlgebraicProduct, is_input=True)
        output_links, output_offset = knowledge_base.matrix(AlgebraicProduct, is_input=False)

        # the following checks that the links between antecedents' memberships (input_links)
        # and the links between rules' activations (output_links) to the consequence layer
        # is correctly constructed and stored in the Mamdani FLC inference engine
        assert torch.isclose(input_links, flc.engine.links['input']).all()
        assert torch.isclose(output_links, flc.engine.links['output']).all()

        # the following checks that the offset matrix is correctly constructed and stored in the
        # Mamdani FLC inference engine; it is added to the steps in the event that a fuzzy set is
        # missing in the input space (input_offset) or the output space (output_offset)
        assert torch.isclose(input_offset, flc.engine.offset['input']).all()
        assert torch.isclose(output_offset, flc.engine.offset['output']).all()

        # check that the antecedents of the Mamdani FLC refers to the input
        # granulation layer (i.e., the fuzzy sets defined in the input space)
        assert (flc.input_granulation.centers == torch.tensor([
            [1.2000, 3.0000, 5.0000, 7.0000],
            [0.2000, 0.6000, 0.9000, 1.2000]
        ])).all()
        assert (flc.input_granulation.sigmas == torch.tensor([
            [0.1000, 0.4000, 0.6000, 0.8000],
            [0.4000, 0.4000, 0.5000, 0.4500]
        ])).all()

        # check that the consequence of the Mamdani FLC inference engine refers to the output
        # granulation layer (i.e., the fuzzy sets defined in the output space)
        # specifically, the centers are used in the Mamdani FLC inference prediction
        assert torch.isclose(flc.engine.consequences.centers, torch.tensor([
            [0.5000, 0.3000, 0.1000, 0.9000],
            [-0.2000, -0.7000, -0.9000, 0.0000]
        ])).all()
        assert torch.isclose(flc.engine.consequences.widths, torch.tensor([
            [0.1000, 0.4000, 0.6000, 0.8000],
            [0.4000, 0.4000, 0.5000, -1.0000]
        ])).all()

        expected_y = torch.tensor(np.array([
            [0.74682458, 1.94174445],
            [0.74682458, 1.94174445],
            [0.74682458, 1.94174445],
            [0.74682458, 1.34428394],
            [0.74682458, 1.94174445]
        ]))
        predicted_y = flc(input_data)

        assert torch.isclose(predicted_y, expected_y).all()
