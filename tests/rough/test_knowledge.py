"""
Tests various aspects of the KnowledgeBase such as whether it is equivalent to another
KnowledgeBase, how it can be reduced, and relative operations such as the relative CORE.
"""
import unittest

from soft.computing.knowledge import KnowledgeBase
from tests.rough.test_relations import example_knowledge_base


class TestEquivalentKnowledge(unittest.TestCase):
    """
    Test that two KnowledgeBase objects are equivalent under the expected circumstances.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 10)])
        self.knowledge_1 = KnowledgeBase()
        self.knowledge_1.set_granules(self.universe)
        self.knowledge_2 = KnowledgeBase()
        self.knowledge_2.set_granules(self.universe)

    def test_equivalent_knowledge(self):
        """
        Test that equivalence between KnowledgeBase holds if they have the same relations.

        Returns:
            None
        """
        for knowledge_base in [self.knowledge_1, self.knowledge_2]:
            example_knowledge_base(knowledge_base, self.universe)

        assert frozenset((self.knowledge_1 / ["R1", "R2", "R3"]).values()) == frozenset(
            (self.knowledge_2 / ["R1", "R2", "R3"]).values()
        )

        assert self.knowledge_1.indiscernibility(
            ["R1", "R2", "R3"]
        ) == self.knowledge_2.indiscernibility(["R1", "R2", "R3"])

    def test_nonequivalent_knowledge(self):
        """
        Test that equivalence between KnowledgeBase does not hold if they do not
        have the same relations.

        Returns:
            None
        """
        for idx, knowledge_base in enumerate([self.knowledge_1, self.knowledge_2]):
            knowledge_base.add_parent_relation(
                "R1", ({"x1", "x3", "x7"}, {"x2", "x4"}, {"x5", "x6", "x8"})
            )
            knowledge_base.add_parent_relation(
                "R2", ({"x1", "x5"}, {"x2", "x6"}, {"x3", "x4", "x7", "x8"})
            )

            if idx == 0:
                knowledge_base.add_parent_relation(
                    "R3", ({"x2", "x7", "x8"}, {"x1", "x3", "x4", "x5", "x6"})
                )

        assert frozenset((self.knowledge_1 / ["R1", "R2", "R3"]).values()) != frozenset(
            (self.knowledge_2 / ["R1", "R2", "R3"]).values()
        )

        with self.assertRaises(
            ValueError
        ):  # exception is thrown since relations are not subset
            _ = self.knowledge_1.indiscernibility(
                ["R1", "R2", "R3"]
            ) != self.knowledge_2.indiscernibility(["R1", "R2", "R3"])

        assert frozenset((self.knowledge_1 / ["R1", "R2", "R3"]).values()) != frozenset(
            (self.knowledge_2 / ["R1", "R2"]).values()
        )

        assert self.knowledge_1.indiscernibility(
            ["R1", "R2", "R3"]
        ) != self.knowledge_2.indiscernibility(["R1", "R2"])


class TestReductionOfKnowledge(unittest.TestCase):
    """
    Test that equivalence classes, indispensability/dispensability, reducts, cores, etc. are
    correctly calculated within the KnowledgeBase.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 10)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.knowledge_base.add_parent_relation(
            "P",
            {
                frozenset({"x1", "x4", "x5"}),
                frozenset({"x2", "x8"}),
                frozenset({"x3"}),
                frozenset({"x6", "x7"}),
            },
        )
        self.knowledge_base.add_parent_relation(
            "Q",
            {
                frozenset({"x1", "x3", "x5"}),
                frozenset({"x6"}),
                frozenset({"x2", "x4", "x7", "x8"}),
            },
        )
        self.knowledge_base.add_parent_relation(
            "R",
            {
                frozenset({"x1", "x5"}),
                frozenset({"x6"}),
                frozenset({"x2", "x7", "x8"}),
                frozenset({"x3", "x4"}),
            },
        )

    def test_equivalence_classes(self):
        """
        Test that equivalence classes of a family of relations is correctly calculated.

        Returns:
            None
        """
        assert self.knowledge_base.indiscernibility({"P", "Q", "R"}) == {
            frozenset({"x1", "x5"}),
            frozenset({"x2", "x8"}),
            frozenset({"x3"}),
            frozenset({"x4"}),
            frozenset({"x6"}),
            frozenset({"x7"}),
        }

    def test_indispensable_p_in_relations(self):
        """
        Test that relation 'P' is indispensable in the family of relations.

        Returns:
            None
        """
        assert not self.knowledge_base.dispensable(
            {"P", "Q", "R"}, "P", mode=self.knowledge_base.indiscernibility
        )
        # what is expected to be returned from IND(Q, R)
        assert self.knowledge_base.indiscernibility({"Q", "R"}) == {
            frozenset({"x1", "x5"}),
            frozenset({"x2", "x7", "x8"}),
            frozenset({"x3"}),
            frozenset({"x4"}),
            frozenset({"x6"}),
        }

    def test_dispensable_q_in_relations(self):
        """
        Test that relation 'Q' is dispensable in the family of relations.

        Returns:
            None
        """
        assert self.knowledge_base.dispensable(
            {"P", "Q", "R"}, "Q", mode=self.knowledge_base.indiscernibility
        )
        # what is expected to be returned from IND(P, R)
        assert self.knowledge_base.indiscernibility(
            {"P", "R"}
        ) == self.knowledge_base.indiscernibility({"P", "Q", "R"})

    def test_dispensable_r_in_relations(self):
        """
        Test that relation 'R' is dispensable in the family of relations.

        Returns:
            None
        """
        assert self.knowledge_base.dispensable(
            {"P", "Q", "R"}, "R", mode=self.knowledge_base.indiscernibility
        )
        # what is expected to be returned from IND(P, R)
        assert self.knowledge_base.indiscernibility(
            {"P", "Q"}
        ) == self.knowledge_base.indiscernibility({"P", "Q", "R"})

    def test_reducts(self):
        """
        Test that the reduct is correctly calculated.

        Returns:
            None
        """
        assert self.knowledge_base.find_reducts({"P", "Q", "R"}) == frozenset(
            {frozenset({"P", "Q"}), frozenset({"P", "R"})}
        )

    def test_core(self):
        """
        Test that the core is correctly calculated.

        Returns:
            None
        """
        assert self.knowledge_base.find_core({"P", "Q", "R"}) == frozenset({"P"})


class TestRelativeReductAndRelativeCore(unittest.TestCase):
    """
    Example 2 on page 36
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 9)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)
        self.knowledge_base.add_parent_relation(
            "P",
            {frozenset({"x1", "x3", "x4", "x5", "x6", "x7"}), frozenset({"x2", "x8"})},
        )
        self.knowledge_base.add_parent_relation(
            "Q",
            {frozenset({"x1", "x3", "x4", "x5"}), frozenset({"x2", "x6", "x7", "x8"})},
        )
        self.knowledge_base.add_parent_relation(
            "R",
            {
                frozenset({"x1", "x5", "x6"}),
                frozenset({"x2", "x7", "x8"}),
                frozenset({"x3", "x4"}),
            },
        )
        self.knowledge_base.add_parent_relation(
            "S",
            {
                frozenset({"x1", "x5", "x6"}),
                frozenset({"x3", "x4"}),
                frozenset({"x2", "x7"}),
                frozenset({"x8"}),
            },
        )

    def test_classification_with_all_relations(self):
        """
        Test that the elements of the universe are correctly classified
        when given a family of relations.

        Returns:
            None
        """
        assert self.knowledge_base.indiscernibility({"P", "Q", "R"}) == frozenset(
            {
                frozenset({"x1", "x5"}),
                frozenset({"x3", "x4"}),
                frozenset({"x2", "x8"}),
                frozenset({"x6"}),
                frozenset({"x7"}),
            }
        )

        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"}, {"S"}
        ) == frozenset({"x1", "x3", "x4", "x5", "x6", "x7"})

    def test_relation_p_is_indispensable(self):
        """
        Test that relation 'P' is S-indispensable.

        Returns:
            None
        """
        equivalence_classes = self.knowledge_base.indiscernibility(
            {"P", "Q", "R"} - {"P"}
        )
        assert equivalence_classes == frozenset(
            {
                frozenset({"x1", "x5"}),
                frozenset({"x3", "x4"}),
                frozenset({"x2", "x7", "x8"}),
                frozenset({"x6"}),
            }
        )

        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"} - {"P"}, {"S"}
        ) == frozenset({"x1", "x3", "x4", "x5", "x6"})
        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"} - {"P"}, {"S"}
        ) != self.knowledge_base.find_relative_positive_region({"P", "Q", "R"}, {"S"})
        # hence, P is S-indispensible in {'P', 'Q', 'R'}
        assert not self.knowledge_base.dispensable(
            {"P", "Q", "R"},
            {"P"},
            relative_to={"S"},
            mode=self.knowledge_base.find_relative_positive_region,
        )

    def test_relation_is_dispensable(self):
        """
        Test that relation 'Q' is S-dispensable.

        Returns:
            None
        """
        equivalence_classes = self.knowledge_base.indiscernibility(
            {"P", "Q", "R"} - {"Q"}
        )
        assert equivalence_classes == frozenset(
            {
                frozenset({"x1", "x5", "x6"}),
                frozenset({"x3", "x4"}),
                frozenset({"x2", "x8"}),
                frozenset({"x7"}),
            }
        )

        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"} - {"Q"}, {"S"}
        ) == frozenset({"x1", "x3", "x4", "x5", "x6", "x7"})
        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"} - {"Q"}, {"S"}
        ) == self.knowledge_base.find_relative_positive_region({"P", "Q", "R"}, {"S"})
        # hence, Q is S-dispensible in {'P', 'Q', 'R'}
        assert self.knowledge_base.dispensable(
            {"P", "Q", "R"},
            {"Q"},
            relative_to={"S"},
            mode=self.knowledge_base.find_relative_positive_region,
        )

    def test_relation_r_is_indispensable(self):
        """
        Test that relation 'R' is S-indispensable.

        Returns:
            None
        """
        equivalence_classes = self.knowledge_base.indiscernibility(
            {"P", "Q", "R"} - {"R"}
        )
        assert equivalence_classes == frozenset(
            {
                frozenset({"x1", "x3", "x4", "x5"}),
                frozenset({"x2", "x8"}),
                frozenset({"x6", "x7"}),
            }
        )

        assert (
            self.knowledge_base.find_relative_positive_region(
                {"P", "Q", "R"} - {"R"}, {"S"}
            )
            == frozenset()
        )
        assert self.knowledge_base.find_relative_positive_region(
            {"P", "Q", "R"} - {"R"}, {"S"}
        ) != self.knowledge_base.find_relative_positive_region({"P", "Q", "R"}, {"S"})
        # hence, R is S-indispensible in {'P', 'Q', 'R'}
        assert not self.knowledge_base.dispensable(
            {"P", "Q", "R"},
            {"R"},
            relative_to={"S"},
            mode=self.knowledge_base.find_relative_positive_region,
        )

    def test_core(self):
        """
        Test the S-core.

        Returns:
            None
        """
        relations = {"P", "Q", "R"}
        # assert self.gg.Q_CORE(relations, self.gg.POS, {'S'}) == frozenset({'P', 'R'})
        assert self.knowledge_base.find_core(relations, relative_to={"S"}) == frozenset(
            {"P", "R"}
        )

    def test_reduct(self):
        """
        Test the S-reduct.

        Returns:
            None
        """
        relations = {"P", "Q", "R"}
        assert self.knowledge_base.find_reducts(
            relations, relative_to={"S"}
        ) == frozenset({frozenset({"P", "R"})})


class TestReductionOfCategories(unittest.TestCase):
    """
    Tests various operations work as intended when involving a family intersection/union.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.universe = frozenset([f"x{i}" for i in range(1, 9)])
        self.knowledge_base = KnowledgeBase()
        self.knowledge_base.set_granules(self.universe)

    def test_family_intersection(self):
        """
        Test that the family intersection works as intended.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x3", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )

        # intersection of the family of sets
        assert self.knowledge_base.family_intersection({"X", "Y", "Z"}) == frozenset(
            {"x1", "x3"}
        )

        # X is indispensable
        assert self.knowledge_base.family_intersection(
            {"X", "Y", "Z"} - {"X"}
        ) == frozenset({"x1", "x3", "x4", "x6"})

        # Y is dispensable
        assert self.knowledge_base.family_intersection(
            {"X", "Y", "Z"} - {"Y"}
        ) == frozenset({"x1", "x3"})

        # Z is dispensable
        assert self.knowledge_base.family_intersection(
            {"X", "Y", "Z"} - {"Z"}
        ) == frozenset({"x1", "x3"})

    def test_family_union(self):
        """
        Test that the family union works as intended.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x2", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )
        self.knowledge_base.add_parent_relation(
            "T", {frozenset({"x1", "x2", "x5", "x7"})}
        )
        # intersection of the family of sets
        assert self.knowledge_base.family_union({"X", "Y", "Z", "T"}) == self.universe
        # X is indispensable
        assert self.knowledge_base.family_union(
            {"X", "Y", "Z", "T"} - {"X"}
        ) == frozenset({"x1", "x2", "x3", "x4", "x5", "x6", "x7"})
        # Y is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"Y"})
            == self.universe
        )
        # Z is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"Z"})
            == self.universe
        )
        # T is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"T"})
            == self.universe
        )

    def test_dispensable(self):
        """
        Test that dispensable works correctly when the family intersection is involved.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x3", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )

        # yields {'x1', 'x3', 'x4', 'x6'}
        assert not self.knowledge_base.dispensable(
            {"X", "Y", "Z"}, "X", mode=self.knowledge_base.family_intersection
        )
        # yields {'x1', 'x3'}
        assert self.knowledge_base.dispensable(
            {"X", "Y", "Z"}, "Y", mode=self.knowledge_base.family_intersection
        )

        # yields {'x1', 'x3'}
        assert self.knowledge_base.dispensable(
            {"X", "Y", "Z"}, "Z", mode=self.knowledge_base.family_intersection
        )

    def test_dependent(self):
        """
        Test that dependency works correctly when the family intersection is involved.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x3", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )
        self.knowledge_base.add_parent_relation(
            "F",
            {
                frozenset({"x1", "x3", "x8"}),
                frozenset({"x1", "x3", "x4", "x5", "x6"}),
                frozenset({"x1", "x3", "x4", "x6", "x7"}),
            },
        )
        assert not self.knowledge_base.independent(
            {"X", "Y", "Z"}, mode=self.knowledge_base.family_intersection
        )

    def test_reducts_and_core(self):
        """
        Additional tests for reducts and cores.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x3", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )
        self.knowledge_base.add_parent_relation(
            "F",
            {
                frozenset({"x1", "x3", "x8"}),
                frozenset({"x1", "x3", "x4", "x5", "x6"}),
                frozenset({"x1", "x3", "x4", "x6", "x7"}),
            },
        )
        assert self.knowledge_base.find_reducts(
            {"X", "Y", "Z"}, mode=self.knowledge_base.family_intersection
        ) == frozenset({frozenset({"X", "Z"}), frozenset({"Y", "X"})})
        assert self.knowledge_base.find_core(
            {"X", "Y", "Z"}, mode=self.knowledge_base.family_intersection
        ) == frozenset({"X"})

    def test_family_union_dispensable(self):
        """
        Test the dispensability of the family union of sets.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x2", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )
        self.knowledge_base.add_parent_relation(
            "T", {frozenset({"x1", "x2", "x5", "x7"})}
        )

        # intersection of the family of sets
        assert self.knowledge_base.family_union({"X", "Y", "Z", "T"}) == self.universe

        # X is indispensable
        assert self.knowledge_base.family_union(
            {"X", "Y", "Z", "T"} - {"X"}
        ) == frozenset({"x1", "x2", "x3", "x4", "x5", "x6", "x7"})
        assert not self.knowledge_base.dispensable(
            {"X", "Y", "Z", "T"}, "X", mode=self.knowledge_base.family_union
        )

        # Y is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"Y"})
            == self.universe
        )
        assert self.knowledge_base.dispensable(
            {"X", "Y", "Z", "T"}, "Y", mode=self.knowledge_base.family_union
        )

        # Z is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"Z"})
            == self.universe
        )
        assert self.knowledge_base.dispensable(
            {"X", "Y", "Z", "T"}, "Z", mode=self.knowledge_base.family_union
        )

        # T is dispensable
        assert (
            self.knowledge_base.family_union({"X", "Y", "Z", "T"} - {"T"})
            == self.universe
        )
        assert self.knowledge_base.dispensable(
            {"X", "Y", "Z", "T"}, "T", mode=self.knowledge_base.family_union
        )

    def test_indispensable(self):
        """
        Test whether "T" is indispensable. Page 40 of the book.

        Returns:
            None
        """
        self.knowledge_base.add_parent_relation("X", {frozenset({"x1", "x3", "x8"})})
        self.knowledge_base.add_parent_relation(
            "Y", {frozenset({"x1", "x3", "x4", "x5", "x6"})}
        )
        self.knowledge_base.add_parent_relation(
            "Z", {frozenset({"x1", "x3", "x4", "x6", "x7"})}
        )
        self.knowledge_base.add_parent_relation("T", {frozenset({"x1", "x3", "x8"})})
        set_f = {"X", "Y", "Z"}

        assert self.knowledge_base.family_intersection(set_f) == frozenset({"x1", "x3"})
        assert not self.knowledge_base.y_dispensable(
            set_f, "T", "X"
        )  # X is T-indispensable
        assert self.knowledge_base.y_dispensable(set_f, "T", "Y")  # Y is T-dispensable
        assert self.knowledge_base.y_dispensable(set_f, "T", "Z")  # Z is T-dispensable

        assert not self.knowledge_base.y_independent(set_f, "Y")
