"""
Test the KnowledgeBase correctly handles fundamental rough set theory operations such as the
calculations of cores, reducts, dispensability, etc.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase


def make_example():
    """
    Make an example that is commonly used between different test scenarios.

    Returns:
        universe of discourse (frozenset), KnowledgeBase
    """
    universe = frozenset(range(1, 9))
    knowledge_base = KnowledgeBase()
    knowledge_base.set_granules(universe)
    knowledge_base.add_parent_relation("a", ({2, 8}, {1, 4, 5}, {3, 6, 7}))
    knowledge_base.add_parent_relation("b", ({1, 3, 5}, {2, 4, 7, 8}, {6}))
    knowledge_base.add_parent_relation("c", ({3, 4, 6}, {2, 7, 8}, {1, 5}))
    knowledge_base.add_parent_relation("d", ({5, 8}, {2, 3, 6, 7}, {1, 4}))
    knowledge_base.add_parent_relation("e", ({1}, {3, 5, 6, 8}, {2, 4, 7}))
    return universe, knowledge_base


class TestKnowledgeRepresentationSystem(unittest.TestCase):
    """
    Test the KnowledgeBase correctly handles various functionality such as cores, reducts, etc.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe, self.knowledge_base = make_example()

    def test_exemplary_partitions(self):
        """
        Test the exemplary partitions of the KnowledgeBase, which are calculated using the
        indiscernibility relation.

        Returns:
            None
        """
        assert self.knowledge_base.indiscernibility("a") == {
            frozenset({8, 2}),
            frozenset({3, 6, 7}),
            frozenset({1, 4, 5}),
        }
        assert self.knowledge_base.indiscernibility("b") == {
            frozenset({8, 2, 4, 7}),
            frozenset({1, 3, 5}),
            frozenset({6}),
        }
        assert self.knowledge_base.indiscernibility({"c", "d"}) == {
            frozenset({3, 6}),
            frozenset({8}),
            frozenset({1}),
            frozenset({5}),
            frozenset({2, 7}),
            frozenset({4}),
        }
        assert self.knowledge_base.indiscernibility({"a", "b", "c"}) == {
            frozenset({8, 2}),
            frozenset({7}),
            frozenset({3}),
            frozenset({1, 5}),
            frozenset({6}),
            frozenset({4}),
        }

    def test_set_approximations(self):
        """
        Test that set approximations with lower, upper, and boundary are correctly calculated.

        Returns:
            None
        """
        set_c, set_x = {"a", "b", "c"}, set(range(1, 6))

        assert self.knowledge_base.lower(set_c, set_x) == frozenset({1, 3, 4, 5})
        assert self.knowledge_base.upper(set_c, set_x) == frozenset({1, 2, 3, 4, 5, 8})
        # unable to decide if 2 or 8 belong to the set set_x or not, using attributes set_c
        assert self.knowledge_base.boundary(set_c, set_x) == frozenset({2, 8})

    def test_dispensability(self):
        """
        Test attribute dispensability.

        Returns:
            None
        """
        set_c = {"a", "b", "c"}

        # the set of attributes set_c are dependent
        assert self.knowledge_base.dependent(set_c, self.knowledge_base.indiscernibility)
        # attributes 'a' and 'b' are indispensable
        assert self.knowledge_base.indispensable(set_c, "a", self.knowledge_base.indiscernibility)
        assert self.knowledge_base.indispensable(set_c, "b", self.knowledge_base.indiscernibility)
        # attribute 'c' is dispensable
        assert self.knowledge_base.dispensable(set_c, "c", self.knowledge_base.indiscernibility)

    def test_reduct(self):
        """
        Test that the reduct is correctly calculated.

        Returns:
            None
        """
        set_c = {"a", "b", "c"}

        # only one reduct in the set set_c
        assert self.knowledge_base.find_reducts(
            set_c, self.knowledge_base.indiscernibility
        ) == frozenset({frozenset({"a", "b"})})

    def test_core(self):
        """
        Test that the core is correctly calculated.

        Returns:
            None
        """
        set_c = {"a", "b", "c"}

        # only one core in the set set_c
        assert self.knowledge_base.find_core(
            set_c, self.knowledge_base.indiscernibility
        ) == frozenset({"a", "b"})

    def test_dependency(self):
        """
        Test that dependency is correctly calculated.

        Returns:
            None
        """
        # since {'a', 'b'} are the reduct & core of set set_c,
        # then we have the dependency: {'a', 'b'} ==> {'c'}
        assert self.knowledge_base.depends_on({"a", "b"}, {"c"})
        assert self.knowledge_base.indiscernibility({"a", "b"}) == {
            frozenset({1, 5}),
            frozenset({2, 8}),
            frozenset({3}),
            frozenset({4}),
            frozenset({6}),
            frozenset({7}),
        }
        assert self.knowledge_base.indiscernibility({"c"}) == {
            frozenset({1, 5}),
            frozenset({2, 7, 8}),
            frozenset({3, 4, 6}),
        }

    def test_attribute_dependency(self):
        """
        Test the dependency between two groups of attributes.

        Returns:
            None
        """
        set_c, set_d = {"a", "b", "c"}, {"d", "e"}
        set_x_1, set_x_2, set_x_3, set_x_4, set_x_5 = {1}, {2, 7}, {3, 6}, {4}, {5, 8}
        set_y_1, set_y_2, set_y_3, set_y_4, set_y_5, set_y_6 = (
            {1, 5},
            {2, 8},
            {3},
            {4},
            {6},
            {7},
        )

        assert self.knowledge_base.indiscernibility(set_d) == {
            frozenset(set_x_1),
            frozenset(set_x_2),
            frozenset(set_x_3),
            frozenset(set_x_4),
            frozenset(set_x_5),
        }

        assert self.knowledge_base.indiscernibility(set_c) == {
            frozenset(set_y_1),
            frozenset(set_y_2),
            frozenset(set_y_3),
            frozenset(set_y_4),
            frozenset(set_y_5),
            frozenset(set_y_6),
        }

        assert self.knowledge_base.lower(set_c, set_x_1) == frozenset()
        assert self.knowledge_base.lower(set_c, set_x_2) == frozenset(set_y_6)
        assert self.knowledge_base.lower(set_c, set_x_3) == frozenset(
            set_y_3.union(set_y_5)
        )
        assert self.knowledge_base.lower(set_c, set_x_4) == frozenset(set_y_4)
        assert self.knowledge_base.lower(set_c, set_x_5) == frozenset()

        # only these elements can be classified into
        # blocks of the partition U / IND(set_d) using set_c
        assert self.knowledge_base.positive_region(set_c, set_d) == frozenset(set_y_3).union(
            set_y_4, set_y_5, set_y_6
        )

        assert self.knowledge_base.partial_depends_on(set_c, set_d) == 0.5

        assert self.knowledge_base.independent_of(set_c, set_d)
        assert self.knowledge_base.indispensable(
            set_c, "a", self.knowledge_base.positive_region, set_d
        )
        assert not self.knowledge_base.dispensable(
            set_c, "a", self.knowledge_base.positive_region, set_d
        )

        assert self.knowledge_base.find_restricted_core(set_c, set_d) == frozenset({"a"})
        assert self.knowledge_base.find_restricted_reducts(set_c, set_d) == frozenset(
            {frozenset({"a", "b"}), frozenset({"a", "c"})}
        )

        # the above means that the following dependencies hold:
        # WARNING: this may be wrong, but the same example says that
        # using set_c we can only classify 4 objects in U / IND(set_d)
        assert self.knowledge_base.partial_depends_on({"a", "b"}, {"d", "e"}) > 0
        assert self.knowledge_base.partial_depends_on({"a", "c"}, {"d", "e"}) > 0

    def test_significance_of_attributes(self):
        """
        Test that the significance of attributes is correctly calculated. The attribute's
        significance is determined by whether the partial dependency changes after its removal.

        Returns:
            None
        """
        set_c, set_d = {"a", "b", "c"}, {"d", "e"}

        assert self.knowledge_base.indiscernibility({"b", "c"}) == {
            frozenset({1, 5}),
            frozenset({2, 7, 8}),
            frozenset({3}),
            frozenset({4}),
            frozenset({6}),
        }
        assert self.knowledge_base.indiscernibility({"a", "c"}) == {
            frozenset({1, 5}),
            frozenset({2, 8}),
            frozenset({3, 6}),
            frozenset({4}),
            frozenset({7}),
        }
        assert self.knowledge_base.indiscernibility({"a", "b"}) == {
            frozenset({1, 5}),
            frozenset({2, 8}),
            frozenset({3}),
            frozenset({4}),
            frozenset({6}),
            frozenset({7}),
        }
        assert self.knowledge_base.indiscernibility({"d", "e"}) == {
            frozenset({1}),
            frozenset({2, 7}),
            frozenset({3, 6}),
            frozenset({4}),
            frozenset({5, 8}),
        }

        assert self.knowledge_base.positive_region(set_c - {"a"}, set_d) == frozenset({3, 4, 6})
        assert self.knowledge_base.positive_region(set_c - {"b"}, set_d) == frozenset({3, 4, 6, 7})
        assert self.knowledge_base.positive_region(set_c - {"c"}, set_d) == frozenset({3, 4, 6, 7})

        # attribute significance is the difference in the partial dependency
        # upon the removal of attributes (pg. 58)
        # attribute 'a' is the most significant
        # (i.e., w/o 'a' we cannot classify object 7 to classes of U / IND(set_d))
        assert (
            self.knowledge_base.partial_depends_on(set_c, set_d)
            - self.knowledge_base.partial_depends_on(set_c - {"a"}, set_d)
            == 0.125
        )
        assert (
            self.knowledge_base.partial_depends_on(set_c, set_d)
            - self.knowledge_base.partial_depends_on(set_c - {"b"}, set_d)
            == 0.0
        )
        assert (
            self.knowledge_base.partial_depends_on(set_c, set_d)
            - self.knowledge_base.partial_depends_on(set_c - {"c"}, set_d)
            == 0.0
        )

        assert not self.knowledge_base.Q_dispensable(
            set_c, set_d, "a"
        )  # attribute 'a' is set_d-indispensable
        assert self.knowledge_base.Q_dispensable(
            set_c, set_d, "b"
        )  # attribute 'b' is set_d-dispensable
        assert self.knowledge_base.Q_dispensable(
            set_c, set_d, "c"
        )  # attribute 'c' is set_d-dispensable

        assert self.knowledge_base.find_restricted_core(set_c, set_d) == frozenset({"a"})
        assert self.knowledge_base.find_restricted_reducts(set_c, set_d) == frozenset(
            {frozenset({"a", "b"}), frozenset({"a", "c"})}
        )
