import unittest

from soft.rough.concepts import Concept
from soft.fuzzy.information.granulation import GranulesGraph


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


class TestReductionOfKnowledge(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(['x{}'.format(i) for i in range(1, 10)])
        self.gg = GranulesGraph(self.universe)
        self.gg.add('P', {frozenset({'x1', 'x4', 'x5'}), frozenset({'x2', 'x8'}),
                          frozenset({'x3'}), frozenset({'x6', 'x7'})})
        self.gg.add('Q', {frozenset({'x1', 'x3', 'x5'}), frozenset({'x6'}),
                          frozenset({'x2', 'x4', 'x7', 'x8'})})
        self.gg.add('R', {frozenset({'x1', 'x5'}), frozenset({'x6'}),
                          frozenset({'x2', 'x7', 'x8'}), frozenset({'x3', 'x4'})})

    def test_equivalence_classes(self):
        assert self.gg.IND({'P', 'Q', 'R'}) == {frozenset({'x1', 'x5'}), frozenset({'x2', 'x8'}), frozenset({'x3'}),
                                                frozenset({'x4'}), frozenset({'x6'}), frozenset({'x7'})}

    def test_indispensable_P_in_relations(self):
        assert self.gg.indispensable({'P', 'Q', 'R'}, 'P')
        assert not self.gg.dispensable({'P', 'Q', 'R'}, 'P')
        # what is expected to be returned from IND(Q, R)
        assert self.gg.IND({'Q', 'R'}) == {frozenset({'x1', 'x5'}), frozenset({'x2', 'x7', 'x8'}), frozenset({'x3'}),
                                           frozenset({'x4'}), frozenset({'x6'})}

    def test_dispensable_Q_in_relations(self):
        assert self.gg.dispensable({'P', 'Q', 'R'}, 'Q')
        assert not self.gg.indispensable({'P', 'Q', 'R'}, 'Q')
        # what is expected to be returned from IND(P, R)
        assert self.gg.IND({'P', 'R'}) == self.gg.IND({'P', 'Q', 'R'})

    def test_dispensable_R_in_relations(self):
        assert self.gg.dispensable({'P', 'Q', 'R'}, 'R')
        assert not self.gg.indispensable({'P', 'Q', 'R'}, 'R')
        # what is expected to be returned from IND(P, R)
        assert self.gg.IND({'P', 'Q'}) == self.gg.IND({'P', 'Q', 'R'})

    def test_reducts(self):
        assert self.gg.RED({'P', 'Q', 'R'}) == frozenset({frozenset({'P', 'Q'}), frozenset({'P', 'R'})})

    def test_core(self):
        assert self.gg.CORE({'P', 'Q', 'R'}) == frozenset({'P'})
