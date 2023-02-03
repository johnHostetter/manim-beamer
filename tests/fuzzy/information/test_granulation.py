import unittest

from soft.computing.graph import KnowledgeBase
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.relation.tnorm import AlgebraicProduct


class TestGranulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.granules = [Gaussian(8), Gaussian(4), Gaussian(6)]

    def test_add_granules(self):
        kb = KnowledgeBase(information=None, config={'granules': self.granules})
        kb.add_fuzzy_granules(self.granules)
        expected_nodes = {
            (0, 1), (0, 7), (2, 4), (1, 2), (0, 4), (2, 1), (0, 0), (1, 1), (0, 3),
            (2, 0), (0, 6), (2, 3), (0, 2), (0, 5), (2, 2), (1, 0), (2, 5), (1, 3)
        }

        assert set([vertex['id'] for vertex in kb.graph.vs.select(variable_notin=[None])]) == expected_nodes

    def test_add_rules(self):
        kb = KnowledgeBase(information=None, config={'granules': self.granules})
        # each tuple is (variable index, term index)
        edges = {
            frozenset({(0, 1), (1, 1)}), frozenset({(1, 2), (2, 3)}), frozenset({(0, 7), (1, 3), (2, 5)})
        }
        kb.add(AlgebraicProduct, edges)

        # only 3 AlgebraicProduct vertices should have been added to the graph; representing the fuzzy logic rule nodes
        algebraic_product_vertices = kb.graph.vs.select(relation_eq=AlgebraicProduct)
        assert len(algebraic_product_vertices) == 3
        assert kb.edges(AlgebraicProduct) == edges  # check the edges have not changed after being added to the graph
