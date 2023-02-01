import unittest

from soft.fuzzy.information.granulation import GranulesGraph


class TestDecisionTable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 9))
        self.gg = GranulesGraph(self.universe)
        self.gg.add('a', ({2, 8}, {1, 4, 5}, {3, 6, 7}))
        self.gg.add('b', ({1, 3, 5}, {2, 4, 7, 8}, {6}))
        self.gg.add('c', ({3, 4, 6}, {2, 7, 8}, {1, 5}))
        self.gg.add('d', ({5, 8}, {2, 3, 6, 7}, {1, 4}))
        self.gg.add('e', ({1}, {3, 5, 6, 8}, {2, 4, 7}))
        self.C, self.D = {'a', 'b', 'c'}, {'d', 'e'}

    def test_rule_consistency(self):
        assert not self.gg.depends_on(self.C, self.D)  # a decision table is consistent iff C ==> D
        equivalence_classes = self.gg.IND(self.C)
        inconsistent_rules = [
            indiscernible_rules for indiscernible_rules in equivalence_classes if len(indiscernible_rules) > 1
        ]
        assert len(inconsistent_rules) == 2  # there should be 2 equivalent groups of rules, each containing 2 rules

    def test_table_decompose(self):
        consistent_rules, inconsistent_rules = self.gg.decompose(self.C, self.D)
        assert consistent_rules == frozenset({3, 4, 6, 7})
        assert inconsistent_rules == frozenset({1, 2, 5, 8})


class TestSimplificationOfDecisionTable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 8))
        self.gg = GranulesGraph(self.universe)
        self.gg.add('a', ({3}, {1, 2, 4, 5}, {6, 7}))
        self.gg.add('b', ({1, 2, 3}, {4, 5, 6}, {7}))
        self.gg.add('c', ({1, 2, 3, 4, 5, 6}, {7}))
        self.gg.add('d', ({2, 3}, {1, 4}, {5, 6, 7}))
        self.gg.add('e', ({3, 4}, {1, 2}, {5, 6, 7}))
        self.C, self.D = {'a', 'b', 'c', 'd'}, {'e'}

    def test_c_is_dispensable(self):
        assert self.gg.dispensable(self.C, 'c', self.gg.IND)
        subset_of_C, = self.gg.Q_RED(self.C, self.D)  # pick the first relative reduct
        assert subset_of_C == frozenset({'b', 'a', 'd'})
        partition_in_each_attribute = self.gg[1]
        family_of_sets = {key: value for key, value in partition_in_each_attribute.items() if key in subset_of_C}
        assert frozenset.intersection(*family_of_sets.values()) == frozenset({1})
        import itertools
        core_attributes, reduct_attributes = {}, {}
        remaining_categories = list(itertools.combinations(subset_of_C, r=len(subset_of_C) - 1))
        for rule_idx in self.universe:
            attr_partitions = self.gg[rule_idx]

            for categories in remaining_categories:
                categories = set(categories)
                family_of_sets = {
                    key: value for key, value in attr_partitions.items() if key in categories
                }

                if not frozenset.intersection(*family_of_sets.values()).issubset(attr_partitions['e']):
                    missing_category, = subset_of_C - categories

                    if rule_idx in core_attributes:
                        core_attributes[rule_idx].append(missing_category)
                    else:
                        core_attributes[rule_idx] = [missing_category]
                else:
                    if rule_idx in reduct_attributes:
                        for attr in family_of_sets.keys():
                            if frozenset.intersection(*(frozenset(family_of_sets.values()) - family_of_sets[attr])).issubset(attr_partitions['e']):
                                reduct_attributes[rule_idx].add(attr)
                            reduct_attributes[rule_idx] = reduct_attributes[rule_idx].union(set(family_of_sets.keys()))
                    else:
                        # reduct_attributes[rule_idx] = set(family_of_sets.keys())
                        reduct_attributes[rule_idx] = set()
                        for attr in family_of_sets.keys():
                            if frozenset.intersection(
                                    *(frozenset(family_of_sets.values()) - family_of_sets[attr])).issubset(
                                    attr_partitions['e']):
                                reduct_attributes[rule_idx].add(attr)
        print()