import unittest

from soft.computing.graph import KnowledgeBase
from soft.rough.constraints import check_if_concepts_intersect


class TestRoughConcept(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = {'a', 'b', 'c', 'd', 'e'}

    def test_check_if_concepts_intersect(self):
        kb = KnowledgeBase(self.universe)
        concept_1, concept_2 = {'a', 'b'}, {'b', 'c'}
        kb.add('R1', (concept_1, concept_2))
        actual_degrees = kb.graph.vs.select(type_eq='rough_sets').outdegree()
        expected_degrees = [0, 0, 1, 1, 2]
        assert sorted(actual_degrees) == expected_degrees
        assert check_if_concepts_intersect(kb)

    def test_add_concepts(self):
        kb = KnowledgeBase(self.universe)
        concept_1, concept_2 = {'a', 'b'}, {'d', 'e'}
        kb.add('R1', (concept_1, concept_2))
        assert not check_if_concepts_intersect(kb)
