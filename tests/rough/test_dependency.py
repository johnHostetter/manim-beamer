"""
Test the various dependencies that can be calculated in the KnowledgeBase such as partial
dependency.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase


class TestDependenciesInKnowledgeBase(unittest.TestCase):
    """
    Test the dependency relations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset(range(1, 9))

    def test_depends_on(self):
        """
        Demonstrates that if Q depends on P, then knowledge Q is
        superfluous within the knowledge base in the sense that
        the knowledge P union Q and P provide the same characterization
        of objects.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(self.universe)
        knowledge_base.add_parent_relation("P", ({1, 5}, {2, 8}, {3}, {4}, {6}, {7}))
        knowledge_base.add_parent_relation("Q", ({1, 5}, {2, 7, 8}, {3, 4, 6}))

        assert knowledge_base.depends_on("P", "Q")
        # partial dependency should equal 1 if depends_on is True
        assert knowledge_base.partial_depends_on("P", "Q") == 1.0
        # the following are all equivalent to the statement above
        assert knowledge_base.indiscernibility({"P", "Q"}) == knowledge_base.indiscernibility({"P"})
        assert knowledge_base.find_restricted_positive_region({"P"}, {"Q"}) == self.universe
        for set_x in knowledge_base / "Q":  # aka 'lower' of IND(P)set_x
            assert knowledge_base.lower_approximation({"P"}, set_x)

    def test_partial_depends_on(self):
        """
        Test the partial dependency relation.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        knowledge_base.set_granules(self.universe)
        set_x_1, set_x_2, set_x_3, set_x_4, set_x_5 = {1}, {2, 7}, {3, 6}, {4}, {5, 8}
        set_y_1, set_y_2, set_y_3, set_y_4, set_y_5, set_y_6 = (
            {1, 5},
            {2, 8},
            {3},
            {4},
            {6},
            {7},
        )
        knowledge_base.add_parent_relation(
            "Q", (set_x_1, set_x_2, set_x_3, set_x_4, set_x_5)
        )
        knowledge_base.add_parent_relation(
            "P", (set_y_1, set_y_2, set_y_3, set_y_4, set_y_5, set_y_6)
        )

        assert knowledge_base.lower_approximation("P", set_x_1) == frozenset()
        assert knowledge_base.lower_approximation("P", set_x_2) == set_y_6
        assert knowledge_base.lower_approximation("P", set_x_3) == set_y_3.union(set_y_5)
        assert knowledge_base.lower_approximation("P", set_x_4) == set_y_4
        assert knowledge_base.lower_approximation("P", set_x_5) == frozenset()
        # only these elements can be classified into blocks of the partition using knowledge P
        assert knowledge_base.find_restricted_positive_region("P", "Q") == set_y_3.union(set_y_4, set_y_5, set_y_6)
        # hence the degree of dependency between Q and P is 0.5
        assert knowledge_base.partial_depends_on("P", "Q") == 0.5
