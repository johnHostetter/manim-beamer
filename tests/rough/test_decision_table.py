"""
Test functions that relation to creating a decision table, simplifying a decision table,
whether rules are consistent, etc.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase
from tests.rough.test_knowledge_representation_system import make_example


class TestDecisionTable(unittest.TestCase):
    """
    Test rule consistency or decision table decomposition.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe, self.knowledge_base = make_example()
        self.set_c, self.set_d = {"a", "b", "c"}, {"d", "e"}

    def test_rule_consistency(self) -> None:
        """
        Test the consistency of rules returns the expected results.

        Returns:
            None
        """
        # a decision table is consistent iff set_c ==> set_d
        assert not self.knowledge_base.depends_on(self.set_c, self.set_d)

        equivalence_classes = self.knowledge_base.indiscernibility(self.set_c)
        inconsistent_rules = [
            indiscernible_rules
            for indiscernible_rules in equivalence_classes
            if len(indiscernible_rules) > 1
        ]

        # there should be 2 equivalent groups of rules, each containing 2 rules
        assert len(inconsistent_rules) == 2

    def test_table_decompose(self) -> None:
        """
        Test the decision table decomposition into two separate sets: rules that are consistent
        and rules that are inconsistent.

        Returns:
            None
        """
        (
            consistent_rules,
            inconsistent_rules,
        ) = self.knowledge_base.decompose_decision_table(self.set_c, self.set_d)
        assert consistent_rules == frozenset({3, 4, 6, 7})
        assert inconsistent_rules == frozenset({1, 2, 5, 8})


class TestSimplificationOfDecisionTable(unittest.TestCase):
    """
    Test the simplification of the decision table, such as whether an attribute is dispensable,
    and check that condition classes are correctly calculated.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 8))
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.knowledge_base.add_parent_relation("a", ({3}, {1, 2, 4, 5}, {6, 7}))
        self.knowledge_base.add_parent_relation("b", ({1, 2, 3}, {4, 5, 6}, {7}))
        self.knowledge_base.add_parent_relation("c", ({1, 2, 3, 4, 5, 6}, {7}))
        self.knowledge_base.add_parent_relation("d", ({2, 3}, {1, 4}, {5, 6, 7}))
        self.knowledge_base.add_parent_relation("e", ({3, 4}, {1, 2}, {5, 6, 7}))
        self.set_c, self.set_d = {"a", "b", "c", "d"}, {"e"}

    def test_c_is_dispensable(self) -> None:
        """
        Test whether attribute 'c' is dispensable.

        Returns:
            None
        """
        assert self.knowledge_base.dispensable(
            self.set_c, "c", mode=self.knowledge_base.indiscernibility
        )

        # pick the first relative reduct
        (subset_of_set_c,) = self.knowledge_base.find_reducts(
            self.set_c, relative_to=self.set_d
        )
        assert subset_of_set_c == frozenset({"b", "a", "d"})
        assert self.knowledge_base.remove_redundant_attributes(
            self.set_c, self.set_d
        ) == frozenset({"b", "a", "d"})

    def test_condition_classes(self) -> None:
        """
        Test that condition classes are correctly calculated or stored.

        Returns:
            None
        """
        partition_in_each_attribute = self.knowledge_base[1]

        # pick the first relative reduct
        (subset_of_set_c,) = self.knowledge_base.find_reducts(
            self.set_c, relative_to=self.set_d
        )
        family_of_sets = {
            key: value
            for key, value in partition_in_each_attribute.items()
            if key in subset_of_set_c
        }
        assert frozenset.intersection(*family_of_sets.values()) == frozenset({1})

    def test_simplify_decision_table(self) -> None:
        """
        Test that decision tables are simplified as expected.

        Returns:
            None
        """
        (
            core_attributes,
            reduct_attributes,
        ) = self.knowledge_base.simplify_decision_table(self.set_c, self.set_d)
        assert core_attributes == {
            1: {"b"},
            2: {"a"},
            3: {"a"},
            4: {"b", "d"},
            5: {"d"},
        }
        assert reduct_attributes == {
            1: {frozenset({"b", "d"}), frozenset({"b", "a"})},
            2: {frozenset({"d", "a"}), frozenset({"b", "a"})},
            3: {frozenset({"a"})},
            4: {frozenset({"b", "d"})},
            5: {frozenset({"d"})},
            6: {frozenset({"a"}), frozenset({"d"})},
            7: {frozenset({"a"}), frozenset({"d"}), frozenset({"b"})},
        }
