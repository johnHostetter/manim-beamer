import unittest

from soft.fuzzy.information.granulation import GranulesGraph
from soft.rough.concepts import Concept


class TestRoughSets(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 9)])
        self.gg = GranulesGraph(self.universe)
        self.E1 = Concept({'x1', 'x4', 'x8'}, self.universe)
        self.E2 = Concept({'x2', 'x5', 'x7'}, self.universe)
        self.E3 = Concept({'x3'}, self.universe)
        self.E4 = Concept({'x6'}, self.universe)
        self.gg.add('R', (self.E1, self.E2, self.E3, self.E4))

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
