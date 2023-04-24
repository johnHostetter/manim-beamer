"""
Tests the fuzzy logic inference engines.
"""
import unittest

import torch

from utils.reproducibility import set_rng, default_configuration
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.logic.rules.creation import Rule
from soft.computing.organize import add_stacked_granule
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.logic.inference.engines import TSKProductInference, TSKMinimumInference

set_rng(0)


def make_test_scenario(configuration):
    """
    Makes a test scenario, with sample data, antecedents, rules, etc.

    Returns:
        Number of output features, consequences (torch.nn.parameter.Parameter), links,
        offset, antecedents_memberships
    """
    input_data = torch.tensor([[1.5409961, -0.2934289],
                               [-2.1787894, 0.56843126],
                               [-1.0845224, -1.3985955],
                               [0.40334684, 0.83802634]])
    antecedents = [
        Gaussian(3, centers=torch.tensor([-1, 0., 1.]), widths=torch.tensor([1., 1., 1.])),
        Gaussian(3, centers=torch.tensor([-1., 0., 1.]), widths=torch.tensor([1., 1., 1.]))]
    rules = [
        Rule(premise=frozenset({(0, 0), (1, 0)}), consequence=frozenset(),
             implication=AlgebraicProduct),
        Rule(premise=frozenset({(0, 0), (1, 1)}), consequence=frozenset(),
             implication=AlgebraicProduct),
        Rule(premise=frozenset({(0, 1), (1, 0)}), consequence=frozenset(),
             implication=AlgebraicProduct),
        Rule(premise=frozenset({(0, 1), (1, 1)}), consequence=frozenset(),
             implication=AlgebraicProduct),
        Rule(premise=frozenset({(0, 1), (1, 2)}), consequence=frozenset(),
             implication=AlgebraicProduct),
    ]

    knowledge_base = expert_design(antecedents, consequents=[], rules=rules, config={})
    links, offset = knowledge_base.links_and_offsets(AlgebraicProduct)
    input_granulation = knowledge_base.graph.vs.find(
        source_eq=add_stacked_granule.__name__)['type']

    out_features = 1
    num_of_consequent_terms = len(rules)
    consequences = torch.nn.parameter.Parameter(torch.zeros(num_of_consequent_terms, out_features))
    consequences.requires_grad = True
    antecedents_memberships = input_granulation(input_data)

    return out_features, consequences, links, offset, antecedents_memberships


class TestFuzzyInference(unittest.TestCase):
    """
    Test the various implementations of fuzzy logic inference.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()

    def test_product_inference_output(self):
        """
        Test the soft.fuzzy.logic.inference.engines.ProductInference class.

        Returns:
            None
        """
        out_features, consequences, links, offset, antecedents_memberships = make_test_scenario(
            configuration=self.config
        )
        product_inference = TSKProductInference(
            out_features=out_features, consequences=consequences,
            links=links, offset=offset)
        actual_output = product_inference.calc_rules_applicability(antecedents_memberships)
        expected_output = torch.tensor([
            [9.52992122e-04, 1.44050468e-03, 5.64775779e-02, 8.53692423e-02, 1.74637646e-02],
            [2.12899314e-02, 1.80385602e-01, 7.41303946e-04, 6.28092951e-03, 7.20215626e-03],
            [8.47027257e-01, 1.40406535e-01, 2.63140523e-01, 4.36191975e-02, 9.78540533e-04],
            [4.75897548e-03, 6.91366485e-02, 2.89834770e-02, 4.21061312e-01, 8.27849315e-01]
        ])
        assert torch.isclose(actual_output, expected_output).all()

    def test_minimum_inference_output(self):
        """
        Test the soft.fuzzy.logic.inference.engines.MinimumInference class.

        Returns:
            None
        """
        out_features, consequences, links, offset, antecedents_memberships = make_test_scenario(
            configuration=self.config
        )
        minimum_inference = TSKMinimumInference(
            out_features=out_features, consequences=consequences,
            links=links, offset=offset)
        actual_output = minimum_inference.calc_rules_applicability(antecedents_memberships)
        expected_output = torch.tensor(
            [[0.00157003, 0.00157003, 0.09304529, 0.09304529, 0.09304529],
             [0.08543695, 0.24918881, 0.00867662, 0.00867662, 0.00867662],
             [0.8531001, 0.1414132, 0.3084521, 0.1414132, 0.00317242],
             [0.034104, 0.13954304, 0.034104, 0.49545035, 0.8498557]])
        assert torch.isclose(actual_output, expected_output).all()
