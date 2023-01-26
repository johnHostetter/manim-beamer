import unittest

from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.information.granulation import GranulesGraph


class TestGranulation(unittest.TestCase):
    def test_add_edges(self):
        graph = GranulesGraph(granules=[Gaussian(8), Gaussian(4), Gaussian(6)])
        expected_nodes = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3),
            (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)
        ]
        assert [vertex['id'] for vertex in graph.network.vs] == expected_nodes

        # each tuple is (variable index, term index)
        graph.add(AlgebraicProduct, [((0, 1), (1, 1)), ((1, 2), (2, 3)), ((0, 7), (1, 3), (2, 5))])

        actual_edges = [
            tuple(graph.network.vs[node]['id'] for node in graph.network.neighbors(edge.target))
            for edge in graph.network.es
        ]
        expected_edges = [
            ((0, 1), (1, 1), 0), ((0, 7), (1, 3), 0), ((0, 7), (2, 5), 0), ((1, 2), (2, 3), 0), ((1, 3), (2, 5), 0)
        ]
        assert actual_edges == expected_edges
