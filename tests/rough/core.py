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


class TestEquivalenceRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])

    def test_get_category(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        gg.add('R1', (concept_1, concept_2, concept_3))
        gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                      Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
        gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                      Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        assert gg / 'R1' == frozenset(
            {frozenset({'x8', 'x6', 'x5'}), frozenset({'x3', 'x7', 'x1'}), frozenset({'x2', 'x4'})}
        )
        assert gg / 'R2' == frozenset(
            {frozenset({'x2', 'x6'}), frozenset({'x1', 'x5'}), frozenset({'x3', 'x8', 'x4', 'x7'})}
        )
        assert gg / 'R3' == frozenset(
            {frozenset({'x2', 'x8', 'x7'}), frozenset({'x1', 'x4', 'x5', 'x3', 'x6'})}
        )

        expected_indexing_result = {
            'R1': frozenset({'x3', 'x1', 'x7'}),
            'R2': frozenset({'x1', 'x5'}),
            'R3': frozenset({'x3', 'x1', 'x4', 'x5', 'x6'})
        }

        assert gg['x1'] == expected_indexing_result

        assert gg['x1']['R1'].intersection(gg['x3']['R2']) == frozenset({'x3', 'x7'})
        assert gg['x2']['R1'].intersection(gg['x2']['R2']) == frozenset({'x2'})
        assert gg['x5']['R1'].intersection(gg['x3']['R2']) == frozenset({'x8'})

        assert gg['x1']['R1'].intersection(gg['x3']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x7'})
        assert gg['x2']['R1'].intersection(gg['x2']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x2'})
        assert gg['x5']['R1'].intersection(gg['x3']['R2']) \
                   .intersection(gg['x2']['R3']) == frozenset({'x8'})

        assert gg['x1']['R1'].union(gg['x2']['R1']) \
               == frozenset({'x1', 'x2', 'x3', 'x4', 'x7'})
        assert gg['x2']['R1'].union(gg['x5']['R1']) \
               == frozenset({'x2', 'x4', 'x5', 'x6', 'x8'})
        assert gg['x1']['R1'].union(gg['x5']['R1']) \
               == frozenset({'x1', 'x3', 'x5', 'x6', 'x7', 'x8'})

        assert gg['x2']['R1'] == frozenset(('x2', 'x4'))
        assert gg['x1']['R2'] == frozenset(('x1', 'x5'))
        assert gg['x2']['R1'].intersection(gg['x1']['R2']) == frozenset()

        assert gg['x1']['R1'] == frozenset(('x1', 'x3', 'x7'))
        assert gg['x2']['R2'] == frozenset(('x2', 'x6'))
        assert gg['x1']['R1'].intersection(gg['x2']['R2']) == frozenset()


class TestEquivalentKnowledge(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])

    def test_equivalent_knowledge(self):
        knowledge_1 = GranulesGraph(self.universe)
        knowledge_2 = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        for gg in [knowledge_1, knowledge_2]:
            gg.add('R1', (concept_1, concept_2, concept_3))
            gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                          Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
            gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                          Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        assert frozenset((knowledge_1 / ['R1', 'R2', 'R3']).values()) == \
               frozenset((knowledge_2 / ['R1', 'R2', 'R3']).values())

        assert knowledge_1.IND(['R1', 'R2', 'R3']) == knowledge_2.IND(['R1', 'R2', 'R3'])

    def test_nonequivalent_knowledge(self):
        knowledge_1 = GranulesGraph(self.universe)
        knowledge_2 = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        for idx, gg in enumerate([knowledge_1, knowledge_2]):
            gg.add('R1', (concept_1, concept_2, concept_3))
            gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                          Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
            if idx == 0:
                gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                              Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        assert frozenset((knowledge_1 / ['R1', 'R2', 'R3']).values()) != \
               frozenset((knowledge_2 / ['R1', 'R2', 'R3']).values())

        with self.assertRaises(ValueError):  # exception is thrown since relations are not subset
            var = knowledge_1.IND(['R1', 'R2', 'R3']) != knowledge_2.IND(['R1', 'R2', 'R3'])

        assert frozenset((knowledge_1 / ['R1', 'R2', 'R3']).values()) != \
               frozenset((knowledge_2 / ['R1', 'R2']).values())

        assert knowledge_1.IND(['R1', 'R2', 'R3']) != knowledge_2.IND(['R1', 'R2'])


class TestIndiscernibilityRelation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])

    def test_indiscernibility(self):
        gg = GranulesGraph(self.universe)
        concept_1, concept_2, concept_3 = Concept({'x1', 'x3', 'x7'}, self.universe), \
            Concept({'x2', 'x4'}, self.universe), Concept({'x5', 'x6', 'x8'}, self.universe)
        gg.add('R1', (concept_1, concept_2, concept_3))
        gg.add('R2', (Concept({'x1', 'x5'}, self.universe), Concept({'x2', 'x6'}, self.universe),
                      Concept({'x3', 'x4', 'x7', 'x8'}, self.universe)))
        gg.add('R3', (Concept({'x2', 'x7', 'x8'}, self.universe),
                      Concept({'x1', 'x3', 'x4', 'x5', 'x6'}, self.universe)))

        gg.IND(['R1', 'R2'])
        print()
