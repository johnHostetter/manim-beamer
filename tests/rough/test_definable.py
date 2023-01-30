import unittest

from soft.rough.concepts import Concept
from soft.fuzzy.information.granulation import GranulesGraph


class TestDefinable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(0, 11)])
        self.gg = GranulesGraph(self.universe)
        self.E1 = Concept({'x0', 'x1'}, self.universe)
        self.E2 = Concept({'x2', 'x6', 'x9'}, self.universe)
        self.E3 = Concept({'x3', 'x5'}, self.universe)
        self.E4 = Concept({'x4', 'x8'}, self.universe)
        self.E5 = Concept({'x7', 'x10'}, self.universe)
        self.gg.add('R', (self.E1, self.E2, self.E3, self.E4, self.E5))

    def test_definable(self):
        X1 = frozenset({'x0', 'x1', 'x4', 'x8'})
        Y1 = frozenset({'x3', 'x4', 'x5', 'x8'})
        Z1 = frozenset({'x2', 'x3', 'x5', 'x6', 'x9'})

        assert self.gg.R_definable('R', X1)
        assert self.gg.R_definable('R', Y1)
        assert self.gg.R_definable('R', Z1)

    def test_roughly_R_definable(self):
        X2 = frozenset({'x0', 'x3', 'x4', 'x5', 'x8', 'x10'})
        Y2 = frozenset({'x1', 'x7', 'x8', 'x10'})
        Z2 = frozenset({'x2', 'x3', 'x4', 'x8'})

        assert self.gg.roughly_R_definable('R', X2)
        assert self.gg.roughly_R_definable('R', Y2)
        assert self.gg.roughly_R_definable('R', Z2)

        # the approximations

        assert self.gg.lower('R', X2) == frozenset(self.E3).union(self.E4)
        assert self.gg.upper('R', X2) == frozenset(self.E1).union(self.E3).union(self.E4).union(self.E5)

        assert self.gg.lower('R', Y2) == frozenset(self.E5)
        assert self.gg.upper('R', Y2) == frozenset(self.E1).union(self.E4).union(self.E5)

        assert self.gg.lower('R', Z2) == frozenset(self.E4)
        assert self.gg.upper('R', Z2) == frozenset(self.E2).union(self.E3).union(self.E4)

        # the boundaries

        assert self.gg.boundary('R', X2) == frozenset(self.E1).union(self.E5)
        assert self.gg.boundary('R', Y2) == frozenset(self.E1).union(self.E4)
        assert self.gg.boundary('R', Z2) == frozenset(self.E2).union(self.E3)

        # the accuracies

        assert self.gg.accuracy('R', X2) == 1 / 2
        assert self.gg.accuracy('R', Y2) == 1 / 3
        assert self.gg.accuracy('R', Z2) == 2 / 7

    def test_externally_R_undefinable(self):
        X3 = frozenset({'x0', 'x1', 'x2', 'x3', 'x4', 'x7'})
        Y3 = frozenset({'x1', 'x2', 'x3', 'x6', 'x8', 'x9', 'x10'})
        Z3 = frozenset({'x0', 'x2', 'x3', 'x4', 'x8', 'x10'})

        assert self.gg.externally_R_undefinable('R', X3)
        assert self.gg.externally_R_undefinable('R', Y3)
        assert self.gg.externally_R_undefinable('R', Z3)

        # the approximations

        assert self.gg.lower('R', X3) == frozenset(self.E1)
        assert self.gg.upper('R', X3) == frozenset(self.universe)

        assert self.gg.lower('R', Y3) == frozenset(self.E2)
        assert self.gg.upper('R', Y3) == frozenset(self.universe)

        assert self.gg.lower('R', Z3) == frozenset(self.E4)
        assert self.gg.upper('R', Z3) == frozenset(self.universe)

        # the boundaries

        assert self.gg.boundary('R', X3) == frozenset(self.E2).union(self.E3).union(self.E4).union(self.E5)
        assert self.gg.boundary('R', Y3) == frozenset(self.E1).union(self.E3).union(self.E4).union(self.E5)
        assert self.gg.boundary('R', Z3) == frozenset(self.E1).union(self.E2).union(self.E3).union(self.E5)

        # the accuracies

        assert self.gg.accuracy('R', X3) == 2 / 11
        assert self.gg.accuracy('R', Y3) == 3 / 11
        assert self.gg.accuracy('R', Z3) == 2 / 11

    def test_internally_R_undefinable(self):
        X4 = frozenset({'x0', 'x2', 'x3'})
        Y4 = frozenset({'x1', 'x2', 'x4', 'x7'})
        Z4 = frozenset({'x2', 'x3', 'x4'})

        assert self.gg.internally_R_undefinable('R', X4)
        assert self.gg.internally_R_undefinable('R', Y4)
        assert self.gg.internally_R_undefinable('R', Z4)

        # the approximations

        assert self.gg.upper('R', X4) == frozenset(self.E1).union(self.E2).union(self.E3)
        assert self.gg.upper('R', Y4) == frozenset(self.E1).union(self.E2).union(self.E4).union(self.E5)
        assert self.gg.upper('R', Z4) == frozenset(self.E2).union(self.E3).union(self.E4)

    def test_totally_R_undefinable(self):
        X5 = frozenset({'x0', 'x2', 'x3', 'x4', 'x7'})
        Y5 = frozenset({'x1', 'x5', 'x6', 'x8', 'x10'})
        Z5 = frozenset({'x0', 'x2', 'x4', 'x5', 'x7'})

        assert self.gg.totally_R_undefinable('R', X5)
        assert self.gg.totally_R_undefinable('R', Y5)
        assert self.gg.totally_R_undefinable('R', Z5)
