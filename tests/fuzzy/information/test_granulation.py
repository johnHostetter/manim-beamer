"""
Test that granulation (i.e., casting precise values into imprecise terms or information granules)
is handled properly. Such features include adding the granules to the KnowledgeBase, as well as
adding rules using the granules to that same KnowledgeBase.
"""
import unittest

from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.logic.rules.creation import Rule
from soft.computing.knowledge import KnowledgeBase
from soft.fuzzy.relation.tnorm import AlgebraicProduct


class TestGranulation(unittest.TestCase):
    """
    Test that granulation is working as intended.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.granules = [Gaussian(8), Gaussian(4), Gaussian(6)]

    def test_add_granules(self) -> None:
        """
        Test that adding granules to a KnowledgeBase will store the granules correctly.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(self.granules)
        knowledge_base.add_fuzzy_granules(self.granules)
        expected_nodes = {
            (0, 1),
            (0, 7),
            (2, 4),
            (1, 2),
            (0, 4),
            (2, 1),
            (0, 0),
            (1, 1),
            (0, 3),
            (2, 0),
            (0, 6),
            (2, 3),
            (0, 2),
            (0, 5),
            (2, 2),
            (1, 0),
            (2, 5),
            (1, 3),
        }

        assert {
            vertex["type"] for vertex in knowledge_base.graph.vs.select(layer_eq=1)
        } == expected_nodes

    def test_add_rules(self) -> None:
        """
        Test that adding rules using the granules will be stored correctly in a KnowledgeBase
        instance.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase(self.granules)
        knowledge_base.add_fuzzy_granules(self.granules)
        # each tuple is (variable index, term index)
        rules = {
            Rule(
                premise=frozenset({(0, 1), (1, 1)}),
                consequence=frozenset(),
                implication=AlgebraicProduct,
            ),
            Rule(
                premise=frozenset({(1, 2), (2, 3)}),
                consequence=frozenset(),
                implication=AlgebraicProduct,
            ),
            Rule(
                premise=frozenset({(0, 7), (1, 3), (2, 5)}),
                consequence=frozenset(),
                implication=AlgebraicProduct,
            ),
        }

        knowledge_base.add_fuzzy_logic_rules(rules)

        # only 3 AlgebraicProduct vertices should have been added to the graph;
        # representing the fuzzy logic rule nodes
        algebraic_product_vertices = knowledge_base.graph.vs.select(
            type_eq=AlgebraicProduct
        )
        assert len(algebraic_product_vertices) == 3
        # check the edges have not changed after being added to the graph
        assert knowledge_base.get_fuzzy_logic_rules() == rules
