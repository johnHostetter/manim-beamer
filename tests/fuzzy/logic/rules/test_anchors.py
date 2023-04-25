"""
Test that fuzzy sets are properly stored, or 'anchored', into a KnowledgeBase.
"""
import unittest

from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.logic.rules.anchors import anchor


class TestAnchor(unittest.TestCase):
    """
    The anchor function is used in children classes
    of FuzzyGraph in order to 'anchor' the fuzzy
    granules in the KnowledgeBase so that they may
    be appropriately queried for systematic design
    processes and construction of FLCs, etc.
    """

    def test_anchor(self):
        """
        Test the 'anchoring' procedure of the KnowledgeBase.

        Returns:
            None
        """
        granules = [Gaussian(8), Gaussian(4), Gaussian(6)]
        expected_anchors = {
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
        }
        expected_variables = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        expected_terms = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5]
        anchors, variables, terms = anchor(granules)
        assert anchors == expected_anchors
        assert variables == expected_variables
        assert terms == expected_terms
