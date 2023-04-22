import unittest

from soft.computing.knowledge import KnowledgeBase


class TestEquivalenceRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])

    def test_get_category(self):
        kb = KnowledgeBase()
        kb.set_granules(self.universe)
        kb.add_parent_relation('R1', ({'x1', 'x3', 'x7'}, {'x2', 'x4'}, {'x5', 'x6', 'x8'}))
        kb.add_parent_relation('R2', ({'x1', 'x5'}, {'x2', 'x6'}, {'x3', 'x4', 'x7', 'x8'}))
        kb.add_parent_relation('R3', ({'x2', 'x7', 'x8'}, {'x1', 'x3', 'x4', 'x5', 'x6'}))

        assert kb / 'R1' == frozenset(
            {frozenset({'x8', 'x6', 'x5'}), frozenset({'x3', 'x7', 'x1'}), frozenset({'x2', 'x4'})}
        )
        assert kb / 'R2' == frozenset(
            {frozenset({'x2', 'x6'}), frozenset({'x1', 'x5'}), frozenset({'x3', 'x8', 'x4', 'x7'})}
        )
        assert kb / 'R3' == frozenset(
            {frozenset({'x2', 'x8', 'x7'}), frozenset({'x1', 'x4', 'x5', 'x3', 'x6'})}
        )

        expected_indexing_result = {
            'R1': frozenset({'x3', 'x1', 'x7'}),
            'R2': frozenset({'x1', 'x5'}),
            'R3': frozenset({'x3', 'x1', 'x4', 'x5', 'x6'})
        }

        assert kb['x1'] == expected_indexing_result

        assert kb['x1']['R1'].intersection(kb['x3']['R2']) == frozenset({'x3', 'x7'})
        assert kb['x2']['R1'].intersection(kb['x2']['R2']) == frozenset({'x2'})
        assert kb['x5']['R1'].intersection(kb['x3']['R2']) == frozenset({'x8'})

        assert kb['x1']['R1'].intersection(kb['x3']['R2']) \
                   .intersection(kb['x2']['R3']) == frozenset({'x7'})
        assert kb['x2']['R1'].intersection(kb['x2']['R2']) \
                   .intersection(kb['x2']['R3']) == frozenset({'x2'})
        assert kb['x5']['R1'].intersection(kb['x3']['R2']) \
                   .intersection(kb['x2']['R3']) == frozenset({'x8'})

        assert kb['x1']['R1'].union(kb['x2']['R1']) \
               == frozenset({'x1', 'x2', 'x3', 'x4', 'x7'})
        assert kb['x2']['R1'].union(kb['x5']['R1']) \
               == frozenset({'x2', 'x4', 'x5', 'x6', 'x8'})
        assert kb['x1']['R1'].union(kb['x5']['R1']) \
               == frozenset({'x1', 'x3', 'x5', 'x6', 'x7', 'x8'})

        assert kb['x2']['R1'] == frozenset(('x2', 'x4'))
        assert kb['x1']['R2'] == frozenset(('x1', 'x5'))
        assert kb['x2']['R1'].intersection(kb['x1']['R2']) == frozenset()

        assert kb['x1']['R1'] == frozenset(('x1', 'x3', 'x7'))
        assert kb['x2']['R2'] == frozenset(('x2', 'x6'))
        assert kb['x1']['R1'].intersection(kb['x2']['R2']) == frozenset()


class TestIndiscernibilityRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])

    def test_indiscernibility(self):
        kb = KnowledgeBase()
        kb.set_granules(self.universe)
        kb.add_parent_relation('R1', ({'x1', 'x3', 'x7'}, {'x2', 'x4'}, {'x5', 'x6', 'x8'}))
        kb.add_parent_relation('R2', ({'x1', 'x5'}, {'x2', 'x6'}, {'x3', 'x4', 'x7', 'x8'}))
        kb.add_parent_relation('R3', ({'x2', 'x7', 'x8'}, {'x1', 'x3', 'x4', 'x5', 'x6'}))

        assert kb.IND(['R1', 'R2']) == {
            frozenset({'x2'}), frozenset({'x5'}), frozenset({'x6'}), frozenset({'x4'}),
            frozenset({'x1'}), frozenset({'x3', 'x7'}), frozenset({'x8'})
        }


class TestRoughEqualityOfSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)
        self.E1 = {'x2', 'x3'}
        self.E2 = {'x1', 'x4', 'x5'}
        self.E3 = {'x6'}
        self.E4 = {'x7', 'x8'}
        self.kb.add_parent_relation('R', (self.E1, self.E2, self.E3, self.E4))

    def test_bottom_R_equal(self):
        X1 = frozenset({'x1', 'x2', 'x3'})
        X2 = frozenset({'x2', 'x3', 'x7'})
        assert self.kb.lower('R', X1) == frozenset(self.E1)
        assert self.kb.lower('R', X2) == frozenset(self.E1)
        assert self.kb.bottom_R_equal('R', X1, X2)

    def test_top_R_equal(self):
        Y1 = frozenset({'x1', 'x2', 'x7'})
        Y2 = frozenset({'x2', 'x3', 'x4', 'x8'})
        assert self.kb.upper('R', Y1) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.kb.upper('R', Y2) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.kb.top_R_equal('R', Y1, Y2)

    def test_R_equal(self):
        Z1 = frozenset({'x1', 'x2', 'x6'})
        Z2 = frozenset({'x3', 'x4', 'x6'})
        assert self.kb.lower('R', Z1) == frozenset(self.E3)
        assert self.kb.lower('R', Z2) == frozenset(self.E3)
        assert self.kb.upper('R', Z1) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.kb.upper('R', Z2) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.kb.R_equal('R', Z1, Z2)


class TestRoughInclusionOfSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)
        self.E1 = {'x2', 'x3'}
        self.E2 = {'x1', 'x4', 'x5'}
        self.E3 = {'x6'}
        self.E4 = {'x7', 'x8'}
        self.kb.add_parent_relation('R', (self.E1, self.E2, self.E3, self.E4))

    def test_bottom_R_included(self):
        X1 = frozenset({'x2', 'x4', 'x6', 'x7'})
        X2 = frozenset({'x2', 'x3', 'x4', 'x6'})
        assert self.kb.lower('R', X1) == frozenset(self.E3)
        assert self.kb.lower('R', X2) == frozenset(self.E1).union(self.E3)
        assert self.kb.bottom_R_included('R', X1, X2)

    def test_top_R_included(self):
        Y1 = frozenset({'x2', 'x3', 'x7'})
        Y2 = frozenset({'x1', 'x2', 'x7'})
        assert self.kb.upper('R', Y1) == frozenset(self.E1).union(self.E4)
        assert self.kb.upper('R', Y2) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.kb.top_R_included('R', Y1, Y2)

    def test_R_included(self):
        Z1 = frozenset({'x2', 'x3'})
        Z2 = frozenset({'x1', 'x2', 'x3', 'x7'})
        assert self.kb.R_included('R', Z1, Z2)
