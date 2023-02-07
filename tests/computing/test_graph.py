import unittest

from soft.computing.knowledge import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_attributes(self):
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 11)])
        self.kb = KnowledgeBase(self.universe)
        self.kb.add_parent_relation('a', ({'x1', 'x2', 'x10'}, {'x4', 'x6', 'x8'}, {'x3'}, {'x5', 'x7'}, {'x9'}))
        self.kb.add_parent_relation('b', ({'x1', 'x3', 'x7'}, {'x2', 'x4'}, {'x5', 'x6', 'x8'}))
        self.kb.add_parent_relation('c', ({'x1', 'x5'}, {'x2', 'x6'}, {'x3', 'x4', 'x7', 'x8'}))
        self.kb.add_parent_relation('d', ({'x2', 'x7', 'x8'}, {'x1', 'x3', 'x4', 'x5', 'x6'}))

        assert self.kb.attributes('x1') == {'a': 0, 'b': 0, 'c': 0, 'd': 1}
        assert self.kb.attributes('x2') == {'a': 0, 'b': 1, 'c': 1, 'd': 0}
        assert self.kb.attributes('x3') == {'a': 2, 'b': 0, 'c': 2, 'd': 1}
        assert self.kb.attributes('x4') == {'a': 1, 'b': 1, 'c': 2, 'd': 1}
        assert self.kb.attributes('x5') == {'a': 3, 'b': 2, 'c': 0, 'd': 1}
        assert self.kb.attributes('x6') == {'a': 1, 'b': 2, 'c': 1, 'd': 1}
        assert self.kb.attributes('x7') == {'a': 3, 'b': 0, 'c': 2, 'd': 0}
        assert self.kb.attributes('x8') == {'a': 1, 'b': 2, 'c': 2, 'd': 0}
        assert self.kb.attributes('x9') == {'a': 4}
        assert self.kb.attributes('x10') == {'a': 0}

    def test_empty_knowledge_base(self):
        self.kb = KnowledgeBase()