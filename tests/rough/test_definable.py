"""
Test the definability (e.g., roughly definable, totally undefinable) of sets
given a family of relations.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase


class TestDefinable(unittest.TestCase):
    """
    Test the various forms of definability return the expected results.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(0, 11)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.set_e_1 = {"x0", "x1"}
        self.set_e_2 = {"x2", "x6", "x9"}
        self.set_e_3 = {"x3", "x5"}
        self.set_e_4 = {"x4", "x8"}
        self.set_e_5 = {"x7", "x10"}
        self.knowledge_base.add_parent_relation(
            "R", (self.set_e_1, self.set_e_2, self.set_e_3, self.set_e_4, self.set_e_5)
        )

    def test_definable(self):
        """
        Test if the sets are defineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_1 = frozenset({"x0", "x1", "x4", "x8"})
        set_y_1 = frozenset({"x3", "x4", "x5", "x8"})
        set_z_1 = frozenset({"x2", "x3", "x5", "x6", "x9"})

        assert type(self.knowledge_base.definable("R", set_x_1)).__name__ == "Definable"
        assert type(self.knowledge_base.definable("R", set_y_1)).__name__ == "Definable"
        assert type(self.knowledge_base.definable("R", set_z_1)).__name__ == "Definable"

    def test_roughly_definable(self):
        """
        Test if the sets are roughly definable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_2 = frozenset({"x0", "x3", "x4", "x5", "x8", "x10"})
        set_y_2 = frozenset({"x1", "x7", "x8", "x10"})
        set_z_2 = frozenset({"x2", "x3", "x4", "x8"})

        assert (
            type(self.knowledge_base.definable("R", set_x_2)).__name__
            == "RoughlyDefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_2)).__name__
            == "RoughlyDefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_2)).__name__
            == "RoughlyDefinable"
        )

        # the approximations

        assert self.knowledge_base.lower_approximation("R", set_x_2) == frozenset(
            self.set_e_3
        ).union(self.set_e_4)
        assert self.knowledge_base.upper_approximation("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)

        assert self.knowledge_base.lower_approximation("R", set_y_2) == frozenset(
            self.set_e_5
        )
        assert self.knowledge_base.upper_approximation("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_4).union(self.set_e_5)

        assert self.knowledge_base.lower_approximation("R", set_z_2) == frozenset(
            self.set_e_4
        )
        assert self.knowledge_base.upper_approximation("R", set_z_2) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4)

        # the boundaries

        assert self.knowledge_base.boundary_region("R", set_x_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_y_2) == frozenset(
            self.set_e_1
        ).union(self.set_e_4)
        assert self.knowledge_base.boundary_region("R", set_z_2) == frozenset(
            self.set_e_2
        ).union(self.set_e_3)

        # the accuracies

        assert self.knowledge_base.accuracy("R", set_x_2) == 1 / 2
        assert self.knowledge_base.accuracy("R", set_y_2) == 1 / 3
        assert self.knowledge_base.accuracy("R", set_z_2) == 2 / 7

    def test_externally_undefinable(self):
        """
        Test if the sets are externally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_3 = frozenset({"x0", "x1", "x2", "x3", "x4", "x7"})
        set_y_3 = frozenset({"x1", "x2", "x3", "x6", "x8", "x9", "x10"})
        set_z_3 = frozenset({"x0", "x2", "x3", "x4", "x8", "x10"})

        assert (
            type(self.knowledge_base.definable("R", set_x_3)).__name__
            == "ExternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_3)).__name__
            == "ExternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_3)).__name__
            == "ExternallyUndefinable"
        )

        # the approximations

        assert self.knowledge_base.lower_approximation("R", set_x_3) == frozenset(
            self.set_e_1
        )
        assert self.knowledge_base.upper_approximation("R", set_x_3) == frozenset(
            self.universe
        )

        assert self.knowledge_base.lower_approximation("R", set_y_3) == frozenset(
            self.set_e_2
        )
        assert self.knowledge_base.upper_approximation("R", set_y_3) == frozenset(
            self.universe
        )

        assert self.knowledge_base.lower_approximation("R", set_z_3) == frozenset(
            self.set_e_4
        )
        assert self.knowledge_base.upper_approximation("R", set_z_3) == frozenset(
            self.universe
        )

        # the boundaries

        assert self.knowledge_base.boundary_region("R", set_x_3) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_y_3) == frozenset(
            self.set_e_1
        ).union(self.set_e_3).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.boundary_region("R", set_z_3) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3).union(self.set_e_5)

        # the accuracies

        assert self.knowledge_base.accuracy("R", set_x_3) == 2 / 11
        assert self.knowledge_base.accuracy("R", set_y_3) == 3 / 11
        assert self.knowledge_base.accuracy("R", set_z_3) == 2 / 11

    def test_internally_undefinable(self):
        """
        Test if the sets are internally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_4 = frozenset({"x0", "x2", "x3"})
        set_y_4 = frozenset({"x1", "x2", "x4", "x7"})
        set_z_4 = frozenset({"x2", "x3", "x4"})

        assert (
            type(self.knowledge_base.definable("R", set_x_4)).__name__
            == "InternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_4)).__name__
            == "InternallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_4)).__name__
            == "InternallyUndefinable"
        )

        # the approximations

        assert self.knowledge_base.upper_approximation("R", set_x_4) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_3)
        assert self.knowledge_base.upper_approximation("R", set_y_4) == frozenset(
            self.set_e_1
        ).union(self.set_e_2).union(self.set_e_4).union(self.set_e_5)
        assert self.knowledge_base.upper_approximation("R", set_z_4) == frozenset(
            self.set_e_2
        ).union(self.set_e_3).union(self.set_e_4)

    def test_totally_undefinable(self):
        """
        Test if the sets are totally undefineable, with respect to the
        given family of relations 'R'.

        Returns:
            None
        """
        set_x_5 = frozenset({"x0", "x2", "x3", "x4", "x7"})
        set_y_5 = frozenset({"x1", "x5", "x6", "x8", "x10"})
        set_z_5 = frozenset({"x0", "x2", "x4", "x5", "x7"})

        assert (
            type(self.knowledge_base.definable("R", set_x_5)).__name__
            == "TotallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_y_5)).__name__
            == "TotallyUndefinable"
        )
        assert (
            type(self.knowledge_base.definable("R", set_z_5)).__name__
            == "TotallyUndefinable"
        )
