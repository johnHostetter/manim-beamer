import unittest

from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.information.granulation import GranulesGraph


class TestGranulation(unittest.TestCase):
    def test_input_granulation(self):
        graph = GranulesGraph(granules=[Gaussian(8), Gaussian(4), Gaussian(6)])
        graph.add(AlgebraicProduct, ((0, 1), (1, 1)), ((1, 2), (2, 3)), ((0, 7), (1, 3), (2, 5)))
        # graph.add(AlgebraicProduct, [(0, 1)])
        # map = GranulesMap(in_features=2, granules_params=[Gaussian(2), Gaussian(2)], membership_function=Gaussian)
        graph.visualize()
