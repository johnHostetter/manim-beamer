import unittest

from soft.computing.knowledge import KnowledgeBase


class TestRoughSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.gg = KnowledgeBase(self.universe)
        self.E1 = {'x1', 'x4', 'x8'}
        self.E2 = {'x2', 'x5', 'x7'}
        self.E3 = {'x3'}
        self.E4 = {'x6'}
        self.gg.add_parent_relation('R', (self.E1, self.E2, self.E3, self.E4))

    def test_equivalence_classes(self):
        assert self.gg / 'R' == frozenset(
            {frozenset(self.E1),
             frozenset(self.E2),
             frozenset(self.E3),
             frozenset(self.E4)}
        )

    def test_lower_approximation(self):
        X1 = frozenset({'x1', 'x4', 'x7'})
        X2 = frozenset({'x2', 'x8'})

        assert self.gg.lower('R', X1) == frozenset()
        assert self.gg.lower('R', X2) == frozenset()
        assert self.gg.lower('R', X1.union(X2)) == frozenset(self.E1)
        assert self.gg.lower('R', X1).union(self.gg.lower('R', X2)).issubset(
            self.gg.lower('R', X1.union(X2)))

    def test_upper_approximation(self):
        Y1 = frozenset({'x1', 'x3', 'x5'})
        Y2 = frozenset({'x2', 'x3', 'x4', 'x6'})

        assert self.gg.upper('R', Y1.intersection(Y2)) == frozenset(self.E3)
        assert self.gg.upper('R', Y1) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.gg.upper('R', Y2) == \
               frozenset(self.E1).union(self.E2).union(self.E3).union(self.E4)
        assert self.gg.upper('R', Y2) == self.universe
        assert self.gg.upper('R', Y1.intersection(Y2)).issubset(
            self.gg.upper('R', Y1).intersection(self.gg.upper('R', Y2)))

    def test_boundary(self):
        X1 = frozenset({'x1', 'x4', 'x5'})
        X2 = frozenset({'x3', 'x5'})
        X3 = frozenset({'x3', 'x6', 'x8'})

        assert self.gg.boundary('R', X1) == frozenset(self.E1).union(self.E2)
        assert self.gg.boundary('R', X2) == frozenset(self.E2)
        assert self.gg.boundary('R', X3) == frozenset(self.E1)

    def test_negative(self):
        X1 = frozenset({'x1', 'x4', 'x5'})
        X2 = frozenset({'x3', 'x5'})
        X3 = frozenset({'x3', 'x6', 'x8'})

        assert self.gg.negative('R', X1) == frozenset({'x3', 'x6'})
        assert self.gg.negative('R', X2) == frozenset(self.E1).union(self.E4)
        assert self.gg.negative('R', X3) == frozenset(self.E2)

    def test_accuracy(self):
        X1 = frozenset({'x1', 'x4', 'x5'})
        X2 = frozenset({'x3', 'x5'})
        X3 = frozenset({'x3', 'x6', 'x8'})

        assert self.gg.accuracy('R', X1) == 0.
        assert self.gg.accuracy('R', X2) == 0.25
        assert self.gg.accuracy('R', X3) == 0.4

    def test_roughness(self):
        X1 = frozenset({'x1', 'x4', 'x5'})
        X2 = frozenset({'x3', 'x5'})
        X3 = frozenset({'x3', 'x6', 'x8'})

        # this should be the complement of accuracy
        assert self.gg.roughness('R', X1) == 1
        assert self.gg.roughness('R', X2) == 0.75
        assert self.gg.roughness('R', X3) == 0.6


class TestApproximationOfClassifications(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.kb = KnowledgeBase(self.universe)
        self.X1 = {'x1', 'x3', 'x5'}
        self.X2 = {'x2', 'x4'}
        self.X3 = {'x6', 'x7', 'x8'}
        self.kb.add_parent_relation('R', (self.X1, self.X2, self.X3))

    def test_classifications_1(self):
        Y1 = frozenset({'x1', 'x2', 'x4'})
        Y2 = frozenset({'x3', 'x5', 'x8'})
        Y3 = frozenset({'x6', 'x7'})

        assert self.kb.lower('R', Y1) == frozenset(self.X2)
        assert self.kb.upper('R', Y2) == frozenset(self.X1).union(self.X3) != self.universe
        assert self.kb.upper('R', Y3) == frozenset(self.X3) != self.universe

    def test_classifications_2(self):
        Z1 = frozenset({'x1', 'x2', 'x6'})
        Z2 = frozenset({'x3', 'x4'})
        Z3 = frozenset({'x5', 'x7', 'x8'})

        assert self.kb.upper('R', Z1) == frozenset(self.X1).union(self.X2).union(self.X3) == self.universe
        assert self.kb.lower('R', Z2) == frozenset()
        assert self.kb.lower('R', Z3) == frozenset()
