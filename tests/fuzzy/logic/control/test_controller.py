"""
Test the ZeroOrderTSK is working as intended, such as its output is correctly calculated.
"""
import unittest

import torch
import numpy as np

from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.logic.controller import ZeroOrderTSK, Mamdani
from utils.reproducibility import set_rng, default_configuration
from examples.fuzzy.offline.supervised.demo_mamdani import toy_mamdani

set_rng(0)


class TestTSK(unittest.TestCase):
    """
    Test the zero-order TSK neuro-fuzzy network.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()

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
            Rule(premise=frozenset({(0, 0), (1, 0)}),
                 consequence=frozenset(), implication=AlgebraicProduct),
            Rule(premise=frozenset({(0, 1), (1, 0)}),
                 consequence=frozenset(), implication=AlgebraicProduct),
            Rule(premise=frozenset({(0, 1), (1, 1)}),
                 consequence=frozenset(), implication=AlgebraicProduct),
            Rule(premise=frozenset({(1, 1), (1, 1)}),
                 consequence=frozenset(), implication=AlgebraicProduct)
        }
        knowledge_base = expert_design(antecedents, consequents=[], rules=rules, config=self.config)

        rule_vertex = knowledge_base.graph.vs.find(type_eq=AlgebraicProduct)
        assert rule_vertex['type'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'type' in rule_vertex.attributes()  # it has 'type' attribute

        rule_vertices = knowledge_base.graph.vs.select(type_eq=AlgebraicProduct)
        assert len(rule_vertices) == len(rules)  # number of rule vertices should equal len(rules)

        # there should be 2 rules that use (1, 1);
        # the last rule has been simplified (redundant mention of condition)
        assert list(knowledge_base[(1, 1)].keys())[0] == AlgebraicProduct
        assert set(knowledge_base[(1, 1)][AlgebraicProduct]) == {
            frozenset({(0, 1), (1, 1)}), frozenset({(1, 1)})
        }

        # the rules we have added should exist how we expected them
        assert knowledge_base.get_fuzzy_logic_rules() == rules

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


class TestMamdani(unittest.TestCase):
    """
    Test the Mamdani neuro-fuzzy network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()
        self.antecedents, self.consequents, self.rules = toy_mamdani()
        self.knowledge_base = expert_design(
            self.antecedents, self.consequents, rules=self.rules, config=self.config)

    def test_mamdani(self):
        """
        Test the Mamdani neuro-fuzzy network.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()

        rule_vertex = self.knowledge_base.graph.vs.find(type_eq=AlgebraicProduct)
        assert rule_vertex['type'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'type' in rule_vertex.attributes()  # it has 'type' attribute

        rule_vertices = self.knowledge_base.graph.vs.select(type_eq=AlgebraicProduct)
        # number of rule vertices should equal len(rules)
        assert len(rule_vertices) == len(self.rules)

        # the rules we have added should exist how we expected them
        assert self.knowledge_base.get_fuzzy_logic_rules() == self.rules

        # check the intra-dimensionality of the input & output spaces are correctly calculated
        assert all(self.knowledge_base.intra_dimensions(True) == np.array([
            term.in_features for term in self.antecedents]))
        assert all(self.knowledge_base.intra_dimensions(False) == np.array([
            term.in_features for term in self.consequents]))
        # the above is required to generate the correct links shape for fuzzy inference

        # check the variable dimensionality of the input & output spaces is correctly calculated
        assert self.knowledge_base.variable_dimensions(True) == len(self.antecedents)
        assert self.knowledge_base.variable_dimensions(False) == len(self.consequents)
        # the above is required to generate the correct links shape for fuzzy inference

        flc = Mamdani(
            out_features=len(self.consequents), knowledge_base=self.knowledge_base,
            learning_rate=1e-3)

        # check that the intermediate structure of the Mamdani FLC is correctly built
        input_links, input_offset = self.knowledge_base.matrix(AlgebraicProduct, is_input=True)
        output_links, output_offset = self.knowledge_base.matrix(AlgebraicProduct, is_input=False)

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
            [5.0000000e-01, -6.9999999e-01],
            [1.7279544e-01, -2.4191362e-01],
            [2.4472545e-03, -5.6169494e-03],
            [2.4864213e-01, -5.3699726e-01],
            [1.3655053e-05, -2.5326384e-05]
        ])).float()
        predicted_y = flc(input_data)

        assert torch.isclose(predicted_y, expected_y).all()
