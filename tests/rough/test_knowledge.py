import unittest

from soft.computing.knowledge import KnowledgeBase


class TestEquivalentKnowledge(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])
        self.knowledge_1 = KnowledgeBase()
        self.knowledge_1.set_granules(self.universe)
        self.knowledge_2 = KnowledgeBase()
        self.knowledge_2.set_granules(self.universe)

    def test_equivalent_knowledge(self):
        for kb in [self.knowledge_1, self.knowledge_2]:
            kb.add_parent_relation('R1', ({'x1', 'x3', 'x7'}, {'x2', 'x4'}, {'x5', 'x6', 'x8'}))
            kb.add_parent_relation('R2', ({'x1', 'x5'}, {'x2', 'x6'}, {'x3', 'x4', 'x7', 'x8'}))
            kb.add_parent_relation('R3', ({'x2', 'x7', 'x8'}, {'x1', 'x3', 'x4', 'x5', 'x6'}))

        assert frozenset((self.knowledge_1 / ['R1', 'R2', 'R3']).values()) == \
               frozenset((self.knowledge_2 / ['R1', 'R2', 'R3']).values())

        assert self.knowledge_1.IND(['R1', 'R2', 'R3']) == self.knowledge_2.IND(['R1', 'R2', 'R3'])

    def test_nonequivalent_knowledge(self):
        for idx, kb in enumerate([self.knowledge_1, self.knowledge_2]):
            kb.add_parent_relation('R1', ({'x1', 'x3', 'x7'}, {'x2', 'x4'}, {'x5', 'x6', 'x8'}))
            kb.add_parent_relation('R2', ({'x1', 'x5'}, {'x2', 'x6'}, {'x3', 'x4', 'x7', 'x8'}))

            if idx == 0:
                kb.add_parent_relation('R3', ({'x2', 'x7', 'x8'}, {'x1', 'x3', 'x4', 'x5', 'x6'}))

        assert frozenset((self.knowledge_1 / ['R1', 'R2', 'R3']).values()) != \
               frozenset((self.knowledge_2 / ['R1', 'R2', 'R3']).values())

        with self.assertRaises(ValueError):  # exception is thrown since relations are not subset
            var = self.knowledge_1.IND(['R1', 'R2', 'R3']) != self.knowledge_2.IND(['R1', 'R2', 'R3'])

        assert frozenset((self.knowledge_1 / ['R1', 'R2', 'R3']).values()) != \
               frozenset((self.knowledge_2 / ['R1', 'R2']).values())

        assert self.knowledge_1.IND(['R1', 'R2', 'R3']) != self.knowledge_2.IND(['R1', 'R2'])


class TestReductionOfKnowledge(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)
        self.kb.add_parent_relation('P', {frozenset({'x1', 'x4', 'x5'}), frozenset({'x2', 'x8'}),
                                          frozenset({'x3'}), frozenset({'x6', 'x7'})})
        self.kb.add_parent_relation('Q', {frozenset({'x1', 'x3', 'x5'}), frozenset({'x6'}),
                                          frozenset({'x2', 'x4', 'x7', 'x8'})})
        self.kb.add_parent_relation('R', {frozenset({'x1', 'x5'}), frozenset({'x6'}),
                                          frozenset({'x2', 'x7', 'x8'}), frozenset({'x3', 'x4'})})

    def test_equivalence_classes(self):
        assert self.kb.IND({'P', 'Q', 'R'}) == {frozenset({'x1', 'x5'}), frozenset({'x2', 'x8'}), frozenset({'x3'}),
                                                frozenset({'x4'}), frozenset({'x6'}), frozenset({'x7'})}

    def test_indispensable_P_in_relations(self):
        assert self.kb.indispensable({'P', 'Q', 'R'}, 'P', func=self.kb.IND)
        assert not self.kb.dispensable({'P', 'Q', 'R'}, 'P', func=self.kb.IND)
        # what is expected to be returned from IND(Q, R)
        assert self.kb.IND({'Q', 'R'}) == {frozenset({'x1', 'x5'}), frozenset({'x2', 'x7', 'x8'}),
                                           frozenset({'x3'}), frozenset({'x4'}), frozenset({'x6'})}

    def test_dispensable_Q_in_relations(self):
        assert self.kb.dispensable({'P', 'Q', 'R'}, 'Q', func=self.kb.IND)
        assert not self.kb.indispensable({'P', 'Q', 'R'}, 'Q', func=self.kb.IND)
        # what is expected to be returned from IND(P, R)
        assert self.kb.IND({'P', 'R'}) == self.kb.IND({'P', 'Q', 'R'})

    def test_dispensable_R_in_relations(self):
        assert self.kb.dispensable({'P', 'Q', 'R'}, 'R', func=self.kb.IND)
        assert not self.kb.indispensable({'P', 'Q', 'R'}, 'R', func=self.kb.IND)
        # what is expected to be returned from IND(P, R)
        assert self.kb.IND({'P', 'Q'}) == self.kb.IND({'P', 'Q', 'R'})

    def test_reducts(self):
        assert self.kb.RED({'P', 'Q', 'R'}, func=self.kb.IND) == frozenset({frozenset({'P', 'Q'}), frozenset({'P', 'R'})})

    def test_core(self):
        assert self.kb.CORE({'P', 'Q', 'R'}, func=self.kb.IND) == frozenset({'P'})


class TestRelativeReductAndRelativeCore(unittest.TestCase):
    """
    Example 2 on page 36
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)
        self.kb.add_parent_relation('P', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6', 'x7'}), frozenset({'x2', 'x8'})})
        self.kb.add_parent_relation('Q', {frozenset({'x1', 'x3', 'x4', 'x5'}), frozenset({'x2', 'x6', 'x7', 'x8'})})
        self.kb.add_parent_relation('R', {frozenset({'x1', 'x5', 'x6'}), frozenset({'x2', 'x7', 'x8'}), frozenset({'x3', 'x4'})})
        self.kb.add_parent_relation('S', {
            frozenset({'x1', 'x5', 'x6'}), frozenset({'x3', 'x4'}), frozenset({'x2', 'x7'}), frozenset({'x8'})})

    def test_classification_with_all_relations(self):
        assert self.kb.IND({'P', 'Q', 'R'}) == frozenset(
            {frozenset({'x1', 'x5'}), frozenset({'x3', 'x4'}), frozenset({'x2', 'x8'}),
             frozenset({'x6'}), frozenset({'x7'})
             })

        assert self.kb.POS({'P', 'Q', 'R'}, {'S'}) == frozenset({'x1', 'x3', 'x4', 'x5', 'x6', 'x7'})

    def test_P_is_S_indispensable(self):
        equivalence_classes = self.kb.IND({'P', 'Q', 'R'} - {'P'})
        assert equivalence_classes == frozenset(
            {frozenset({'x1', 'x5'}), frozenset({'x3', 'x4'}), frozenset({'x2', 'x7', 'x8'}), frozenset({'x6'})})

        assert self.kb.POS({'P', 'Q', 'R'} - {'P'}, {'S'}) == frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})
        assert self.kb.POS({'P', 'Q', 'R'} - {'P'}, {'S'}) != self.kb.POS({'P', 'Q', 'R'}, {'S'})
        # hence, P is S-indispensible in {'P', 'Q', 'R'}
        assert self.kb.indispensable({'P', 'Q', 'R'}, {'P'}, self.kb.POS, {'S'})
        assert not self.kb.dispensable({'P', 'Q', 'R'}, {'P'}, self.kb.POS, {'S'})

    def test_Q_is_S_dispensable(self):
        equivalence_classes = self.kb.IND({'P', 'Q', 'R'} - {'Q'})
        assert equivalence_classes == frozenset(
            {frozenset({'x1', 'x5', 'x6'}), frozenset({'x3', 'x4'}), frozenset({'x2', 'x8'}), frozenset({'x7'})})

        assert self.kb.POS({'P', 'Q', 'R'} - {'Q'}, {'S'}) == frozenset({'x1', 'x3', 'x4', 'x5', 'x6', 'x7'})
        assert self.kb.POS({'P', 'Q', 'R'} - {'Q'}, {'S'}) == self.kb.POS({'P', 'Q', 'R'}, {'S'})
        # hence, Q is S-dispensible in {'P', 'Q', 'R'}
        assert self.kb.dispensable({'P', 'Q', 'R'}, {'Q'}, self.kb.POS, {'S'})
        assert not self.kb.indispensable({'P', 'Q', 'R'}, {'Q'}, self.kb.POS, {'S'})

    def test_R_is_S_indispensable(self):
        equivalence_classes = self.kb.IND({'P', 'Q', 'R'} - {'R'})
        assert equivalence_classes == frozenset(
            {frozenset({'x1', 'x3', 'x4', 'x5'}), frozenset({'x2', 'x8'}), frozenset({'x6', 'x7'})})

        assert self.kb.POS({'P', 'Q', 'R'} - {'R'}, {'S'}) == frozenset()
        assert self.kb.POS({'P', 'Q', 'R'} - {'R'}, {'S'}) != self.kb.POS({'P', 'Q', 'R'}, {'S'})
        # hence, R is S-indispensible in {'P', 'Q', 'R'}
        assert self.kb.indispensable({'P', 'Q', 'R'}, {'R'}, self.kb.POS, {'S'})
        assert not self.kb.dispensable({'P', 'Q', 'R'}, {'R'}, self.kb.POS, {'S'})

    def test_S_core(self):
        relations = {'P', 'Q', 'R'}
        # assert self.gg.Q_CORE(relations, self.gg.POS, {'S'}) == frozenset({'P', 'R'})
        assert self.kb.Q_CORE(relations, {'S'}) == frozenset({'P', 'R'})

    def test_S_reduct(self):
        relations = {'P', 'Q', 'R'}
        assert self.kb.Q_RED(relations, {'S'}) == frozenset({frozenset({'P', 'R'})})


class TestReductionOfCategories(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)

    def test_family_intersection(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        # intersection of the family of sets
        assert self.kb.family_intersection({'X', 'Y', 'Z'}) == frozenset({'x1', 'x3'})
        # X is indispensable
        assert self.kb.family_intersection({'X', 'Y', 'Z'} - {'X'}) == frozenset({'x1', 'x3', 'x4', 'x6'})
        # Y is dispensable
        assert self.kb.family_intersection({'X', 'Y', 'Z'} - {'Y'}) == frozenset({'x1', 'x3'})
        # Z is dispensable
        assert self.kb.family_intersection({'X', 'Y', 'Z'} - {'Z'}) == frozenset({'x1', 'x3'})

    def test_family_union(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x2', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        self.kb.add_parent_relation('T', {frozenset({'x1', 'x2', 'x5', 'x7'})})
        # intersection of the family of sets
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'}) == self.universe
        # X is indispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'X'}) == frozenset(
            {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'})
        # Y is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'Y'}) == self.universe
        # Z is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'Z'}) == self.universe
        # T is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'T'}) == self.universe

    def test_dispensable(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})

        assert not self.kb.dispensable({'X', 'Y', 'Z'}, 'X',
                                       func=self.kb.family_intersection)  # yields {'x1', 'x3', 'x4', 'x6'}
        assert self.kb.dispensable({'X', 'Y', 'Z'}, 'Y', func=self.kb.family_intersection)  # yields {'x1', 'x3'}
        assert self.kb.dispensable({'X', 'Y', 'Z'}, 'Z', func=self.kb.family_intersection)  # yields {'x1', 'x3'}

    def test_dependent(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        self.kb.add_parent_relation('F', {frozenset({'x1', 'x3', 'x8'}), frozenset({'x1', 'x3', 'x4', 'x5', 'x6'}),
                                          frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        assert self.kb.dependent({'X', 'Y', 'Z'}, func=self.kb.family_intersection)
        assert not self.kb.independent({'X', 'Y', 'Z'}, func=self.kb.family_intersection)

    def test_reducts_and_core(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        self.kb.add_parent_relation('F', {frozenset({'x1', 'x3', 'x8'}), frozenset({'x1', 'x3', 'x4', 'x5', 'x6'}),
                                          frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        assert self.kb.RED({'X', 'Y', 'Z'}, func=self.kb.family_intersection) == frozenset(
            {frozenset({'X', 'Z'}), frozenset({'Y', 'X'})})
        assert self.kb.CORE({'X', 'Y', 'Z'}, func=self.kb.family_intersection) == frozenset({'X'})

    def test_family_union_dispensable(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x2', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        self.kb.add_parent_relation('T', {frozenset({'x1', 'x2', 'x5', 'x7'})})

        # intersection of the family of sets
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'}) == self.universe

        # X is indispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'X'}) == frozenset(
            {'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'})
        assert not self.kb.dispensable({'X', 'Y', 'Z', 'T'}, 'X', func=self.kb.family_union)

        # Y is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'Y'}) == self.universe
        assert self.kb.dispensable({'X', 'Y', 'Z', 'T'}, 'Y', func=self.kb.family_union)

        # Z is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'Z'}) == self.universe
        assert self.kb.dispensable({'X', 'Y', 'Z', 'T'}, 'Z', func=self.kb.family_union)

        # T is dispensable
        assert self.kb.family_union({'X', 'Y', 'Z', 'T'} - {'T'}) == self.universe
        assert self.kb.dispensable({'X', 'Y', 'Z', 'T'}, 'T', func=self.kb.family_union)

    def test_T_indispensable(self):
        self.kb.add_parent_relation('X', {frozenset({'x1', 'x3', 'x8'})})
        self.kb.add_parent_relation('Y', {frozenset({'x1', 'x3', 'x4', 'x5', 'x6'})})
        self.kb.add_parent_relation('Z', {frozenset({'x1', 'x3', 'x4', 'x6', 'x7'})})
        self.kb.add_parent_relation('T', {frozenset({'x1', 'x3', 'x8'})})
        F = {'X', 'Y', 'Z'}

        assert self.kb.family_intersection(F) == frozenset({'x1', 'x3'})
        assert not self.kb.Y_dispensable(F, 'T', 'X')  # X is T-indispensable
        assert self.kb.Y_dispensable(F, 'T', 'Y')  # Y is T-dispensable
        assert self.kb.Y_dispensable(F, 'T', 'Z')  # Z is T-dispensable

        assert not self.kb.Y_independent(F, 'Y')
