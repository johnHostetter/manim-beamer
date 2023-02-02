import unittest

from soft.computing.graph import KnowledgeBase
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.relation.tnorm import AlgebraicProduct


class TestGranulation(unittest.TestCase):
    def test_add_edges(self):
        graph = KnowledgeBase(granules=[Gaussian(8), Gaussian(4), Gaussian(6)])
        expected_nodes = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3),
            (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)
        ]
        assert [vertex['id'] for vertex in graph.network.vs] == expected_nodes

        # each tuple is (variable index, term index)
        edges = [((0, 1), (1, 1)), ((1, 2), (2, 3)), ((0, 7), (1, 3), (2, 5))]
        graph.add(AlgebraicProduct, edges)

        # only 3 AlgebraicProduct vertices should have been added to the graph; representing the fuzzy logic rule nodes
        algebraic_product_vertices = graph.network.vs.select(relation_eq=AlgebraicProduct)
        assert len(algebraic_product_vertices) == 3

        assert graph.edges(AlgebraicProduct) == edges  # check the edges have not changed after being added to the graph
