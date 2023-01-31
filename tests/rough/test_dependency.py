import unittest

from soft.fuzzy.information.granulation import GranulesGraph


class TestDependenciesInKnowledgeBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 9))

    def test_depends_on(self):
        """
        Demonstrates that if Q depends on P, then knowledge Q is
        superfluous within the knowledge base in the sense that
        the knowledge P union Q and P provide the same characterization
        of objects.

        Returns:

        """
        self.gg = GranulesGraph(self.universe)
        self.gg.add('P', ({1, 5}, {2, 8}, {3}, {4}, {6}, {7}))
        self.gg.add('Q', ({1, 5}, {2, 7, 8}, {3, 4, 6}))

        assert self.gg.depends_on('P', 'Q')
        # partial dependency should equal 1 if depends_on is True
        assert self.gg.partial_depends_on('P', 'Q') == 1.0
        # the following are all equivalent to the statement above
        assert self.gg.IND({'P', 'Q'}) == self.gg.IND({'P'})
        assert self.gg.POS({'P'}, {'Q'}) == self.universe
        for X in self.gg / 'Q':  # aka 'lower' of IND(P)X
            assert self.gg.lower({'P'}, X)

    def test_partial_depends_on(self):
        self.gg = GranulesGraph(self.universe)
        X1, X2, X3, X4, X5 = {1}, {2, 7}, {3, 6}, {4}, {5, 8}
        Y1, Y2, Y3, Y4, Y5, Y6 = {1, 5}, {2, 8}, {3}, {4}, {6}, {7}
        self.gg.add('Q', (X1, X2, X3, X4, X5))
        self.gg.add('P', (Y1, Y2, Y3, Y4, Y5, Y6))

        assert self.gg.lower('P', X1) == frozenset()
        assert self.gg.lower('P', X2) == Y6
        assert self.gg.lower('P', X3) == Y3.union(Y5)
        assert self.gg.lower('P', X4) == Y4
        assert self.gg.lower('P', X5) == frozenset()
        # only these elements can be classified into blocks of the partition using knowledge P
        assert self.gg.POS('P', 'Q') == Y3.union(Y4, Y5, Y6)
        # hence the degree of dependency between Q and P is 0.5
        assert self.gg.partial_depends_on('P', 'Q') == 0.5
