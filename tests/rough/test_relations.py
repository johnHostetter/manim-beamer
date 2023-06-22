"""
Test the equivalence relation, indiscernibility, rough equality, and rough inclusion of sets.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase


def example_knowledge_base(knowledge_base: KnowledgeBase, universe: frozenset) -> None:
    """
    Apply granules and relations in-place to an example KnowledgeBase object. Page 4 of the book.

    Args:
        knowledge_base: The KnowledgeBase object to apply granules and relations to.
        universe: The universe of discourse.

    Returns:
        None
    """
    knowledge_base.set_granules(universe)
    knowledge_base.add_parent_relation(
        "R1", ({"x1", "x3", "x7"}, {"x2", "x4"}, {"x5", "x6", "x8"})
    )
    knowledge_base.add_parent_relation(
        "R2", ({"x1", "x5"}, {"x2", "x6"}, {"x3", "x4", "x7", "x8"})
    )
    knowledge_base.add_parent_relation(
        "R3", ({"x2", "x7", "x8"}, {"x1", "x3", "x4", "x5", "x6"})
    )


class TestEquivalenceRelation(unittest.TestCase):
    """
    Test the equivalence relation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 9)])

    def test_get_category(self) -> None:
        """
        Test that categories are correctly stored by equivalence relation(s).

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        example_knowledge_base(knowledge_base, self.universe)

        assert knowledge_base / "R1" == frozenset(
            {
                frozenset({"x8", "x6", "x5"}),
                frozenset({"x3", "x7", "x1"}),
                frozenset({"x2", "x4"}),
            }
        )
        assert knowledge_base / "R2" == frozenset(
            {
                frozenset({"x2", "x6"}),
                frozenset({"x1", "x5"}),
                frozenset({"x3", "x8", "x4", "x7"}),
            }
        )
        assert knowledge_base / "R3" == frozenset(
            {frozenset({"x2", "x8", "x7"}), frozenset({"x1", "x4", "x5", "x3", "x6"})}
        )

        expected_indexing_result = {
            "R1": frozenset({"x3", "x1", "x7"}),
            "R2": frozenset({"x1", "x5"}),
            "R3": frozenset({"x3", "x1", "x4", "x5", "x6"}),
        }

        assert knowledge_base["x1"] == expected_indexing_result

        assert knowledge_base["x1"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ) == frozenset({"x3", "x7"})
        assert knowledge_base["x2"]["R1"].intersection(
            knowledge_base["x2"]["R2"]
        ) == frozenset({"x2"})
        assert knowledge_base["x5"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ) == frozenset({"x8"})

        assert knowledge_base["x1"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x7"})
        assert knowledge_base["x2"]["R1"].intersection(
            knowledge_base["x2"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x2"})
        assert knowledge_base["x5"]["R1"].intersection(
            knowledge_base["x3"]["R2"]
        ).intersection(knowledge_base["x2"]["R3"]) == frozenset({"x8"})

        assert knowledge_base["x1"]["R1"].union(
            knowledge_base["x2"]["R1"]
        ) == frozenset({"x1", "x2", "x3", "x4", "x7"})
        assert knowledge_base["x2"]["R1"].union(
            knowledge_base["x5"]["R1"]
        ) == frozenset({"x2", "x4", "x5", "x6", "x8"})
        assert knowledge_base["x1"]["R1"].union(
            knowledge_base["x5"]["R1"]
        ) == frozenset({"x1", "x3", "x5", "x6", "x7", "x8"})

        assert knowledge_base["x2"]["R1"] == frozenset(("x2", "x4"))
        assert knowledge_base["x1"]["R2"] == frozenset(("x1", "x5"))
        assert (
                knowledge_base["x2"]["R1"].intersection(knowledge_base["x1"]["R2"])
                == frozenset()
        )

        assert knowledge_base["x1"]["R1"] == frozenset(("x1", "x3", "x7"))
        assert knowledge_base["x2"]["R2"] == frozenset(("x2", "x6"))
        assert (
                knowledge_base["x1"]["R1"].intersection(knowledge_base["x2"]["R2"])
                == frozenset()
        )


class TestIndiscernibilityRelation(unittest.TestCase):
    """
    Test the indiscernibility relation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 10)])

    def test_indiscernibility(self) -> None:
        """
        Test the indiscernibility relation calculates as expected on page 5 of the book.

        Returns:
            None
        """
        knowledge_base = KnowledgeBase()
        example_knowledge_base(knowledge_base, self.universe)

        assert knowledge_base.indiscernibility(["R1", "R2"]) == {
            frozenset({"x2"}),
            frozenset({"x5"}),
            frozenset({"x6"}),
            frozenset({"x4"}),
            frozenset({"x1"}),
            frozenset({"x3", "x7"}),
            frozenset({"x8"}),
        }


class TestRoughEqualityOfSets(unittest.TestCase):
    """
    Test the rough equality of sets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 9)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.set_e_1 = {"x2", "x3"}
        self.set_e_2 = {"x1", "x4", "x5"}
        self.set_e_3 = {"x6"}
        self.set_e_4 = {"x7", "x8"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4)
        )

    def test_bottom_rough_equal(self):
        """
        Test the rough bottom equality of sets is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x1", "x2", "x3"})
        set_x_2 = frozenset({"x2", "x3", "x7"})
        assert self.knowledge_base.lower_approximation("R", set_x_1) == frozenset(self.set_e_1)
        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(self.set_e_1)
        assert self.knowledge_base.bottom_r_equal("R", set_x_1, set_x_2)

    def test_top_rough_equal(self):
        """
        Test the rough top equality of sets is correctly calculated.

        Returns:
            None
        """
        set_y_1 = frozenset({"x1", "x2", "x7"})
        set_y_2 = frozenset({"x2", "x3", "x4", "x8"})
        assert self.knowledge_base.upper_approximation("R", set_y_1) == frozenset(
            self.set_e_1).union(
            self.set_e_2
        ).union(self.set_e_4)
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1).union(
            self.set_e_2
        ).union(self.set_e_4)
        assert self.knowledge_base.top_r_equal("R", set_y_1, set_y_2)

    def test_rough_equal(self):
        """
        Test the rough equality of sets is correctly calculated.

        Returns:
            None
        """
        set_z_1 = frozenset({"x1", "x2", "x6"})
        set_z_2 = frozenset({"x3", "x4", "x6"})
        assert self.knowledge_base.lower_approximation("R", set_z_1) == frozenset(self.set_e_3)
        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_z_1) == frozenset(
            self.set_e_1).union(
            self.set_e_2
        ).union(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_z_2) == frozenset(
            self.set_e_1).union(
            self.set_e_2
        ).union(self.set_e_3)
        assert self.knowledge_base.r_equal("R", set_z_1, set_z_2)


class TestRoughInclusionOfSets(unittest.TestCase):
    """
    Test the rough inclusion of sets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 9)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.set_e_1 = {"x2", "x3"}
        self.set_e_2 = {"x1", "x4", "x5"}
        self.set_e_3 = {"x6"}
        self.set_e_4 = {"x7", "x8"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4)
        )

    def test_bottom_rough_included(self):
        """
        Test the rough bottom inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_x_1 = frozenset({"x2", "x4", "x6", "x7"})
        set_x_2 = frozenset({"x2", "x3", "x4", "x6"})
        assert self.knowledge_base.lower_approximation("R", set_x_1) == frozenset(self.set_e_3)
        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(
            self.set_e_1).union(
            self.set_e_3
        )
        assert self.knowledge_base.bottom_r_included("R", set_x_1, set_x_2)

    def test_top_rough_included(self):
        """
        Test the rough top inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_y_1 = frozenset({"x2", "x3", "x7"})
        set_y_2 = frozenset({"x1", "x2", "x7"})
        assert self.knowledge_base.upper_approximation("R", set_y_1) == frozenset(
            self.set_e_1).union(
            self.set_e_4
        )
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1).union(
            self.set_e_2
        ).union(self.set_e_4)
        assert self.knowledge_base.top_r_included("R", set_y_1, set_y_2)

    def test_rough_included(self):
        """
        Test the rough inclusion of sets is correctly calculated.

        Returns:
            None
        """
        set_z_1 = frozenset({"x2", "x3"})
        set_z_2 = frozenset({"x1", "x2", "x3", "x7"})
        assert self.knowledge_base.r_included("R", set_z_1, set_z_2)
