"""
Test properties of KnowledgeBase, such as how attributes of observations or granules are stored.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase


class TestKnowledgeBase(unittest.TestCase):
    """
    Test the KnowledgeBase class such as checking data attributes are correctly added.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_attributes(self) -> None:
        """
        Test that attributes, and their corresponding references (i.e., relations in KnowledgeBase),
        are appropriately added to the Knowledgebase graph.

        Returns:
            None
        """
        universe = frozenset([f"x{i}" for i in range(1, 11)])
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(universe)
        # group up data points that share the same value for the respective attribute
        attribute_groupings = {
            # for example, 'x1', 'x2', 'x10' have the same value for attribute 'a'
            "a": (
                {"x1", "x2", "x10"},
                {"x4", "x6", "x8"},
                {"x3"},
                {"x5", "x7"},
                {"x9"},
            ),
            # also, 'x2', 'x4' have the same value for attribute 'b'
            "b": ({"x1", "x3", "x7"}, {"x2", "x4"}, {"x5", "x6", "x8"}),
            # and so on
            "c": ({"x1", "x5"}, {"x2", "x6"}, {"x3", "x4", "x7", "x8"}),
            "d": ({"x2", "x7", "x8"}, {"x1", "x3", "x4", "x5", "x6"}),
        }
        knowledge_base.add_parent_relation("a", attribute_groupings["a"])
        knowledge_base.add_parent_relation("b", attribute_groupings["b"])
        knowledge_base.add_parent_relation("c", attribute_groupings["c"])
        knowledge_base.add_parent_relation("d", attribute_groupings["d"])

        # TODO: Fix this; attributes are no longer stored
        # assert knowledge_base.attributes('x1') == {'a': 0, 'b': 0, 'c': 0, 'd': 1}
        # assert knowledge_base.attributes('x2') == {'a': 0, 'b': 1, 'c': 1, 'd': 0}
        # assert knowledge_base.attributes('x3') == {'a': 2, 'b': 0, 'c': 2, 'd': 1}
        # assert knowledge_base.attributes('x4') == {'a': 1, 'b': 1, 'c': 2, 'd': 1}
        # assert knowledge_base.attributes('x5') == {'a': 3, 'b': 2, 'c': 0, 'd': 1}
        # assert knowledge_base.attributes('x6') == {'a': 1, 'b': 2, 'c': 1, 'd': 1}
        # assert knowledge_base.attributes('x7') == {'a': 3, 'b': 0, 'c': 2, 'd': 0}
        # assert knowledge_base.attributes('x8') == {'a': 1, 'b': 2, 'c': 2, 'd': 0}
        # assert knowledge_base.attributes('x9') == {'a': 4}
        # assert knowledge_base.attributes('x10') == {'a': 0}

    def test_empty_knowledge_base(self) -> None:
        """
        Test that if we create a KnowledgeBase with no arguments, that it is indeed empty.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        assert len(knowledge_base.graph.vs) == 0
        assert len(knowledge_base.graph.es) == 0
