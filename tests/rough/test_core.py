import unittest

from soft.rough.concepts import Concept
from soft.fuzzy.information.granulation import GranulesGraph
from soft.rough.constraints import check_if_concepts_intersect


class TestRoughConcept(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = {'a', 'b', 'c', 'd', 'e'}

    def test_check_if_concept_empty(self):
        with self.assertRaises(ValueError):
            Concept(set(), self.universe)

    def test_check_if_concepts_intersect(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2 = Concept({'a', 'b'}, self.universe), Concept({'b', 'c'}, self.universe)
        gg.add('R1', (concept_1, concept_2))
        actual_degrees = gg.network.vs.select(type_eq='rough_sets').outdegree()
        expected_degrees = [0, 0, 1, 1, 2]
        assert sorted(actual_degrees) == expected_degrees
        assert check_if_concepts_intersect(gg)

    def test_add_concepts(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2 = Concept({'a', 'b'}, self.universe), Concept({'d', 'e'}, self.universe)
        gg.add('R1', (concept_1, concept_2))
        assert not check_if_concepts_intersect(gg)
