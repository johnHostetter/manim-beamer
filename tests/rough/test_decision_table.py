import unittest

from soft.computing.graph import KnowledgeBase


class TestDecisionTable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 9))
        self.kb = KnowledgeBase(self.universe)
        self.kb.add('a', ({2, 8}, {1, 4, 5}, {3, 6, 7}))
        self.kb.add('b', ({1, 3, 5}, {2, 4, 7, 8}, {6}))
        self.kb.add('c', ({3, 4, 6}, {2, 7, 8}, {1, 5}))
        self.kb.add('d', ({5, 8}, {2, 3, 6, 7}, {1, 4}))
        self.kb.add('e', ({1}, {3, 5, 6, 8}, {2, 4, 7}))
        self.C, self.D = {'a', 'b', 'c'}, {'d', 'e'}

    def test_rule_consistency(self):
        assert not self.kb.depends_on(self.C, self.D)  # a decision table is consistent iff C ==> D
        equivalence_classes = self.kb.IND(self.C)
        inconsistent_rules = [
            indiscernible_rules for indiscernible_rules in equivalence_classes if len(indiscernible_rules) > 1
        ]
        assert len(inconsistent_rules) == 2  # there should be 2 equivalent groups of rules, each containing 2 rules

    def test_table_decompose(self):
        consistent_rules, inconsistent_rules = self.kb.decompose_decision_table(self.C, self.D)
        assert consistent_rules == frozenset({3, 4, 6, 7})
        assert inconsistent_rules == frozenset({1, 2, 5, 8})


class TestSimplificationOfDecisionTable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 8))
        self.kb = KnowledgeBase(self.universe)
        self.kb.add('a', ({3}, {1, 2, 4, 5}, {6, 7}))
        self.kb.add('b', ({1, 2, 3}, {4, 5, 6}, {7}))
        self.kb.add('c', ({1, 2, 3, 4, 5, 6}, {7}))
        self.kb.add('d', ({2, 3}, {1, 4}, {5, 6, 7}))
        self.kb.add('e', ({3, 4}, {1, 2}, {5, 6, 7}))
        self.C, self.D = {'a', 'b', 'c', 'd'}, {'e'}

    def test_c_is_dispensable(self):
        assert self.kb.dispensable(self.C, 'c', self.kb.IND)
        subset_of_C, = self.kb.Q_RED(self.C, self.D)  # pick the first relative reduct
        assert subset_of_C == frozenset({'b', 'a', 'd'})
        assert self.kb.remove_redundant_attributes(self.C, self.D) == frozenset({'b', 'a', 'd'})

    def test_condition_classes(self):
        partition_in_each_attribute = self.kb[1]
        subset_of_C, = self.kb.Q_RED(self.C, self.D)  # pick the first relative reduct
        family_of_sets = {key: value for key, value in partition_in_each_attribute.items() if key in subset_of_C}
        assert frozenset.intersection(*family_of_sets.values()) == frozenset({1})

    def test_simplify_decision_table(self):
        core_attributes, reduct_attributes = self.kb.simplify_decision_table(self.C, self.D)
        assert core_attributes == {1: {'b'}, 2: {'a'}, 3: {'a'}, 4: {'b', 'd'}, 5: {'d'}}
        assert reduct_attributes == {
            1: {frozenset({'b', 'd'}), frozenset({'b', 'a'})}, 2: {frozenset({'d', 'a'}), frozenset({'b', 'a'})},
            3: {frozenset({'a'})}, 4: {frozenset({'b', 'd'})}, 5: {frozenset({'d'})},
            6: {frozenset({'a'}), frozenset({'d'})}, 7: {frozenset({'a'}), frozenset({'d'}), frozenset({'b'})}
        }
