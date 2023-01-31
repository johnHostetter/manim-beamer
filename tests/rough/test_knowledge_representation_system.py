import unittest

from soft.fuzzy.information.granulation import GranulesGraph


class TestKnowledgeRepresentationSystem(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 9))
        self.gg = GranulesGraph(self.universe)
        self.gg.add('a', ({2, 8}, {1, 4, 5}, {3, 6, 7}))
        self.gg.add('b', ({1, 3, 5}, {2, 4, 7, 8}, {6}))
        self.gg.add('c', ({3, 4, 6}, {2, 7, 8}, {1, 5}))
        self.gg.add('d', ({5, 8}, {2, 3, 6, 7}, {1, 4}))
        self.gg.add('e', ({1}, {3, 5, 6, 8}, {2, 4, 7}))

    def test_exemplary_partitions(self):
        assert self.gg.IND('a') == {frozenset({8, 2}), frozenset({3, 6, 7}), frozenset({1, 4, 5})}
        assert self.gg.IND('b') == {frozenset({8, 2, 4, 7}), frozenset({1, 3, 5}), frozenset({6})}
        assert self.gg.IND({'c', 'd'}) == {
            frozenset({3, 6}), frozenset({8}), frozenset({1}), frozenset({5}), frozenset({2, 7}), frozenset({4})
        }
        assert self.gg.IND({'a', 'b', 'c'}) == {
            frozenset({8, 2}), frozenset({7}), frozenset({3}), frozenset({1, 5}), frozenset({6}), frozenset({4})
        }

    def test_set_approximations(self):
        C, X = {'a', 'b', 'c'}, set(range(1, 6))

        assert self.gg.lower(C, X) == frozenset({1, 3, 4, 5})
        assert self.gg.upper(C, X) == frozenset({1, 2, 3, 4, 5, 8})
        # unable to decide if 2 or 8 belong to the set X or not, using attributes C
        assert self.gg.boundary(C, X) == frozenset({2, 8})

    def test_dispensability(self):
        C = {'a', 'b', 'c'}

        # the set of attributes C are dependent
        assert self.gg.dependent(C, self.gg.IND)
        # attributes 'a' and 'b' are indispensable
        assert self.gg.indispensable(C, 'a', self.gg.IND)
        assert self.gg.indispensable(C, 'b', self.gg.IND)
        # attribute 'c' is dispensable
        assert self.gg.dispensable(C, 'c', self.gg.IND)

    def test_reduct(self):
        C = {'a', 'b', 'c'}

        # only one reduct in the set C
        assert self.gg.RED(C, self.gg.IND) == frozenset({frozenset({'a', 'b'})})

    def test_core(self):
        C = {'a', 'b', 'c'}

        # only one core in the set C
        assert self.gg.CORE(C, self.gg.IND) == frozenset({'a', 'b'})

    def test_dependency(self):
        # since {'a', 'b'} are the reduct & core of set C, then we have the dependency: {'a', 'b'} ==> {'c'}
        assert self.gg.depends_on({'a', 'b'}, {'c'})
        assert self.gg.IND({'a', 'b'}) == {
            frozenset({1, 5}), frozenset({2, 8}), frozenset({3}), frozenset({4}), frozenset({6}), frozenset({7})
        }
        assert self.gg.IND({'c'}) == {frozenset({1, 5}), frozenset({2, 7, 8}), frozenset({3, 4, 6})}

    def test_attribute_dependency(self):
        C, D = {'a', 'b', 'c'}, {'d', 'e'}
        X1, X2, X3, X4, X5 = {1}, {2, 7}, {3, 6}, {4}, {5, 8}
        Y1, Y2, Y3, Y4, Y5, Y6 = {1, 5}, {2, 8}, {3}, {4}, {6}, {7}

        assert self.gg.IND(D) == {
            frozenset(X1), frozenset(X2), frozenset(X3), frozenset(X4), frozenset(X5)
        }

        assert self.gg.IND(C) == {
            frozenset(Y1), frozenset(Y2), frozenset(Y3), frozenset(Y4), frozenset(Y5), frozenset(Y6)
        }

        assert self.gg.lower(C, X1) == frozenset()
        assert self.gg.lower(C, X2) == frozenset(Y6)
        assert self.gg.lower(C, X3) == frozenset(Y3.union(Y5))
        assert self.gg.lower(C, X4) == frozenset(Y4)
        assert self.gg.lower(C, X5) == frozenset()

        # only these elements can be classified into blocks of the partition U / IND(D) using C
        assert self.gg.POS(C, D) == frozenset(Y3).union(Y4, Y5, Y6)

        assert self.gg.partial_depends_on(C, D) == 0.5

        assert self.gg.independent_of(C, D)
        assert not self.gg.Q_dispensable(C, D, 'a')

        # TODO: add relative reducts and cores