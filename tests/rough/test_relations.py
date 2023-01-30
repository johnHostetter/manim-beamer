import unittest

from soft.rough.concepts import Concept
from soft.fuzzy.information.granulation import GranulesGraph


class TestEquivalenceRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])

    def test_get_category(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        gg.add('R1', (concept_1, concept_2, concept_3))
        gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                      Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
        gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                      Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        assert gg / 'R1' == frozenset(
            {frozenset({'x8', 'x6', 'x5'}), frozenset({'x3', 'x7', 'x1'}), frozenset({'x2', 'x4'})}
        )
        assert gg / 'R2' == frozenset(
            {frozenset({'x2', 'x6'}), frozenset({'x1', 'x5'}), frozenset({'x3', 'x8', 'x4', 'x7'})}
        )
        assert gg / 'R3' == frozenset(
            {frozenset({'x2', 'x8', 'x7'}), frozenset({'x1', 'x4', 'x5', 'x3', 'x6'})}
        )

        expected_indexing_result = {
            'R1': frozenset({'x3', 'x1', 'x7'}),
            'R2': frozenset({'x1', 'x5'}),
            'R3': frozenset({'x3', 'x1', 'x4', 'x5', 'x6'})
        }

        assert gg['x1'] == expected_indexing_result

        assert gg['x1']['R1'].intersection(gg['x3']['R2']) == frozenset({'x3', 'x7'})
        assert gg['x2']['R1'].intersection(gg['x2']['R2']) == frozenset({'x2'})
        assert gg['x5']['R1'].intersection(gg['x3']['R2']) == frozenset({'x8'})

        assert gg['x1']['R1'].intersection(gg['x3']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x7'})
        assert gg['x2']['R1'].intersection(gg['x2']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x2'})
        assert gg['x5']['R1'].intersection(gg['x3']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x8'})

        assert gg['x1']['R1'].union(gg['x2']['R1']) \
               == frozenset({'x1', 'x2', 'x3', 'x4', 'x7'})
        assert gg['x2']['R1'].union(gg['x5']['R1']) \
               == frozenset({'x2', 'x4', 'x5', 'x6', 'x8'})
        assert gg['x1']['R1'].union(gg['x5']['R1']) \
               == frozenset({'x1', 'x3', 'x5', 'x6', 'x7', 'x8'})

        assert gg['x2']['R1'] == frozenset(('x2', 'x4'))
        assert gg['x1']['R2'] == frozenset(('x1', 'x5'))
        assert gg['x2']['R1'].intersection(gg['x1']['R2']) == frozenset()

        assert gg['x1']['R1'] == frozenset(('x1', 'x3', 'x7'))
        assert gg['x2']['R2'] == frozenset(('x2', 'x6'))
        assert gg['x1']['R1'].intersection(gg['x2']['R2']) == frozenset()


class TestIndiscernibilityRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])

    def test_indiscernibility(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        gg.add('R1', (concept_1, concept_2, concept_3))
        gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                      Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
        gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                      Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        assert gg.IND(['R1', 'R2']) == {
            frozenset({'x2'}), frozenset({'x5'}), frozenset({'x6'}), frozenset({'x4'}),
            frozenset({'x1'}), frozenset({'x3', 'x7'}), frozenset({'x8'})
        }


class TestRoughEqualityOfSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.gg = GranulesGraph(self.universe)
        self.E1 = Concept({'x2', 'x3'}, self.universe)
        self.E2 = Concept({'x1', 'x4', 'x5'}, self.universe)
        self.E3 = Concept({'x6'}, self.universe)
        self.E4 = Concept({'x7', 'x8'}, self.universe)
        self.gg.add('R', (self.E1, self.E2, self.E3, self.E4))

    def test_bottom_R_equal(self):
        X1 = frozenset({'x1', 'x2', 'x3'})
        X2 = frozenset({'x2', 'x3', 'x7'})
        assert self.gg.lower('R', X1) == frozenset(self.E1)
        assert self.gg.lower('R', X2) == frozenset(self.E1)
        assert self.gg.bottom_R_equal('R', X1, X2)

    def test_top_R_equal(self):
        Y1 = frozenset({'x1', 'x2', 'x7'})
        Y2 = frozenset({'x2', 'x3', 'x4', 'x8'})
        assert self.gg.upper('R', Y1) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.gg.upper('R', Y2) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.gg.top_R_equal('R', Y1, Y2)

    def test_R_equal(self):
        Z1 = frozenset({'x1', 'x2', 'x6'})
        Z2 = frozenset({'x3', 'x4', 'x6'})
        assert self.gg.lower('R', Z1) == frozenset(self.E3)
        assert self.gg.lower('R', Z2) == frozenset(self.E3)
        assert self.gg.upper('R', Z1) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.gg.upper('R', Z2) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.gg.R_equal('R', Z1, Z2)


class TestRoughInclusionOfSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.gg = GranulesGraph(self.universe)
        self.E1 = Concept({'x2', 'x3'}, self.universe)
        self.E2 = Concept({'x1', 'x4', 'x5'}, self.universe)
        self.E3 = Concept({'x6'}, self.universe)
        self.E4 = Concept({'x7', 'x8'}, self.universe)
        self.gg.add('R', (self.E1, self.E2, self.E3, self.E4))

    def test_bottom_R_included(self):
        X1 = frozenset({'x2', 'x4', 'x6', 'x7'})
        X2 = frozenset({'x2', 'x3', 'x4', 'x6'})
        assert self.gg.lower('R', X1) == frozenset(self.E3)
        assert self.gg.lower('R', X2) == frozenset(self.E1).union(self.E3)
        assert self.gg.bottom_R_included('R', X1, X2)

    def test_top_R_included(self):
        Y1 = frozenset({'x2', 'x3', 'x7'})
        Y2 = frozenset({'x1', 'x2', 'x7'})
        assert self.gg.upper('R', Y1) == frozenset(self.E1).union(self.E4)
        assert self.gg.upper('R', Y2) == frozenset(self.E1).union(self.E2).union(self.E4)
        assert self.gg.top_R_included('R', Y1, Y2)

    def test_R_included(self):
        Z1 = frozenset({'x2', 'x3'})
        Z2 = frozenset({'x1', 'x2', 'x3', 'x7'})
        assert self.gg.R_included('R', Z1, Z2)
