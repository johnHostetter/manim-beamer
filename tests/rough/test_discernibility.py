import unittest

from soft.computing.knowledge import KnowledgeBase


class TestDiscernibility(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 6))
        self.kb = KnowledgeBase()
        self.kb.set_granules(self.universe)
        self.kb.add_parent_relation('a', ({1}, {2, 3, 5}, {4}))
        self.kb.add_parent_relation('b', ({3}, {1, 4, 5}, {2}))
        self.kb.add_parent_relation('c', ({2, 4, 5}, {3}, {1}))
        self.kb.add_parent_relation('d', ({1, 3}, {4}, {2, 5}))

    def test_discernibility_matrix(self):
        relations = {'a', 'b', 'c', 'd'}

        assert self.kb.discernibility_matrix(relations) == {
            frozenset({1, 2}): {'c', 'b', 'a', 'd'}, frozenset({1, 3}): {'c', 'b', 'a'},
            frozenset({1, 4}): {'c', 'a', 'd'}, frozenset({1, 5}): {'c', 'a', 'd'}, frozenset({2, 3}): {'c', 'b', 'd'},
            frozenset({2, 4}): {'b', 'a', 'd'}, frozenset({2, 5}): {'b'}, frozenset({3, 4}): {'c', 'b', 'a', 'd'},
            frozenset({3, 5}): {'c', 'b', 'd'}, frozenset({4, 5}): {'a', 'd'}
        }

        assert self.kb.minimum_discernibility_matrix(relations) == {
            frozenset({1, 2}): frozenset({'b'}), frozenset({1, 3}): frozenset({'b'}),
            frozenset({1, 4}): frozenset({'a', 'd'}), frozenset({1, 5}): frozenset({'a', 'd'}),
            frozenset({2, 3}): frozenset({'b'}), frozenset({2, 4}): frozenset({'b'}),
            frozenset({2, 5}): frozenset({'b'}), frozenset({3, 4}): frozenset({'b'}),
            frozenset({3, 5}): frozenset({'b'}), frozenset({4, 5}): frozenset({'a', 'd'})
         }

    def test_discernibility_matrix_on_decision_attribute(self):
        relations = {'a', 'b', 'c'}

        assert self.kb.discernibility_matrix(relations, 'd') == {
            frozenset({1, 2}): {'c', 'b', 'a'}, frozenset({1, 4}): {'c', 'a'}, frozenset({1, 5}): {'c', 'a'},
            frozenset({2, 3}): {'c', 'b'}, frozenset({2, 4}): {'b', 'a'}, frozenset({3, 4}): {'c', 'b', 'a'},
            frozenset({3, 5}): {'c', 'b'}, frozenset({4, 5}): {'a'}
        }

        assert self.kb.minimum_discernibility_matrix(relations, decision_attributes='d') == {
            frozenset({1, 2}): {'a'}, frozenset({1, 4}): {'a'}, frozenset({1, 5}): {'a'},
            frozenset({2, 3}): {'c', 'b'}, frozenset({2, 4}): {'a'}, frozenset({3, 4}): {'a'},
            frozenset({3, 5}): {'c', 'b'}, frozenset({4, 5}): {'a'}
        }
