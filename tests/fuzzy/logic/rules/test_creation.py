"""
Test the various mechanisms in which a fuzzy logic rule can be created.
"""
import unittest

from utils.reproducibility import default_configuration
from soft.computing.design import expert_design
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from examples.fuzzy.offline.supervised.demo_mamdani import toy_mamdani


class TestFuzzyLogicRule(unittest.TestCase):
    """
    Test the operations and functions of a fuzzy logic rule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = default_configuration()

    def test_constructor(self):
        """
        Test that a (Mamdani) fuzzy logic rule is correctly created.

        Returns:
            None
        """
        # the first variable, second term with the second variable, second term and the third
        # variable with the first term is the input
        antecedents = {frozenset({(0, 1), (1, 1), (2, 0)})}
        # the 4th variable and second term is the recommended output
        consequents = {frozenset({(3, 1)})}
        rule = Rule(
            premise=antecedents, consequence=consequents, implication=AlgebraicProduct
        )
        assert rule.premise == antecedents
        assert rule.consequence == consequents
        assert rule.implication == AlgebraicProduct

    def test_add_mamdani_rules_to_knowledge_base(self):
        """
        Test that adding Mamdani fuzzy logic rules to a KnowledgeBase object does not break things.

        Returns:
            None
        """
        antecedents, consequents, rules = toy_mamdani()
        knowledge_base = expert_design(
            antecedents, consequents, rules, config=self.config
        )
        assert len(knowledge_base.graph.vs.select(layer_eq="Rule")) == len(rules)
        assert knowledge_base.get_fuzzy_logic_rules() == rules
