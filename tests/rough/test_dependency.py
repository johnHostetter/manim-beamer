import unittest

from soft.computing.knowledge import KnowledgeBase


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
        self.kb = KnowledgeBase(self.universe)
        self.kb.add_parent_relation('P', ({1, 5}, {2, 8}, {3}, {4}, {6}, {7}))
        self.kb.add_parent_relation('Q', ({1, 5}, {2, 7, 8}, {3, 4, 6}))

        assert self.kb.depends_on('P', 'Q')
        # partial dependency should equal 1 if depends_on is True
        assert self.kb.partial_depends_on('P', 'Q') == 1.0
        # the following are all equivalent to the statement above
        assert self.kb.IND({'P', 'Q'}) == self.kb.IND({'P'})
        assert self.kb.POS({'P'}, {'Q'}) == self.universe
        for X in self.kb / 'Q':  # aka 'lower' of IND(P)X
            assert self.kb.lower({'P'}, X)

    def test_partial_depends_on(self):
        self.kb = KnowledgeBase(self.universe)
        X1, X2, X3, X4, X5 = {1}, {2, 7}, {3, 6}, {4}, {5, 8}
        Y1, Y2, Y3, Y4, Y5, Y6 = {1, 5}, {2, 8}, {3}, {4}, {6}, {7}
        self.kb.add_parent_relation('Q', (X1, X2, X3, X4, X5))
        self.kb.add_parent_relation('P', (Y1, Y2, Y3, Y4, Y5, Y6))

        assert self.kb.lower('P', X1) == frozenset()
        assert self.kb.lower('P', X2) == Y6
        assert self.kb.lower('P', X3) == Y3.union(Y5)
        assert self.kb.lower('P', X4) == Y4
        assert self.kb.lower('P', X5) == frozenset()
        # only these elements can be classified into blocks of the partition using knowledge P
        assert self.kb.POS('P', 'Q') == Y3.union(Y4, Y5, Y6)
        # hence the degree of dependency between Q and P is 0.5
        assert self.kb.partial_depends_on('P', 'Q') == 0.5
