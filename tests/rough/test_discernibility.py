import unittest
import itertools

from soft.fuzzy.information.granulation import GranulesGraph


class TestDiscernibility(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 6))
        self.gg = GranulesGraph(self.universe)
        self.gg.add('a', ({1}, {2, 3, 5}, {4}))
        self.gg.add('b', ({3}, {1, 4, 5}, {2}))
        self.gg.add('c', ({2, 4, 5}, {3}, {1}))
        self.gg.add('d', ({1, 3}, {4}, {2, 5}))

    def test_discernibility_matrix(self):
        relations = {'a', 'b', 'c', 'd'}

        assert self.gg.discernibility_matrix(relations) == {
            frozenset({1, 2}): {'c', 'b', 'a', 'd'}, frozenset({1, 3}): {'c', 'b', 'a'},
            frozenset({1, 4}): {'c', 'a', 'd'}, frozenset({1, 5}): {'c', 'a', 'd'}, frozenset({2, 3}): {'c', 'b', 'd'},
            frozenset({2, 4}): {'b', 'a', 'd'}, frozenset({2, 5}): {'b'}, frozenset({3, 4}): {'c', 'b', 'a', 'd'},
            frozenset({3, 5}): {'c', 'b', 'd'}, frozenset({4, 5}): {'a', 'd'}
        }

        assert self.gg.minimum_discernibility_matrix(relations) == {
            frozenset({1, 2}): frozenset({'b'}), frozenset({1, 3}): frozenset({'b'}),
            frozenset({1, 4}): frozenset({'a', 'd'}), frozenset({1, 5}): frozenset({'a', 'd'}),
            frozenset({2, 3}): frozenset({'b'}), frozenset({2, 4}): frozenset({'b'}),
            frozenset({2, 5}): frozenset({'b'}), frozenset({3, 4}): frozenset({'b'}),
            frozenset({3, 5}): frozenset({'b'}), frozenset({4, 5}): frozenset({'a', 'd'})
         }

    def test_discernibility_matrix_on_decision_attribute(self):
        relations = {'a', 'b', 'c', 'd'}

        pass