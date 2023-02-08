import unittest

import torch

from soft.computing.design import SelfOrganize
from soft.fuzzy.graph.threads import ComponentThread

"""
The following algorithms are eligible for self-organizing neuro-fuzzy networks.
"""

from soft.fuzzy.online.unsupervised.cluster.ecm import ECM
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP
from soft.fuzzy.logic.rules.creation import wang_mendel_method as WM
from soft.fuzzy.offline.unsupervised.cluster.empirical import Empirical as EFS


class TestSelfOrganize(unittest.TestCase):
    """
    The self-organizing process can be thought as
    a Knowledge Base (KB) constructing another KB.
    However, it passes the relevant components
    needed to call the expert design process when
    it has finished, to conclude the construction
    of the KB.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_empty_self_organize_kb(self):
        so = SelfOrganize()
        assert len(so.graph.vs) == 0  # the Self-Organize Knowledge Base was initialized with no vertices
        assert len(so.graph.es) == 0  # the Self-Organize Knowledge Base was initialized with no edges

        assert len(so.kb.graph.vs) == 0  # the Knowledge Base was initialized with no vertices
        assert len(so.kb.graph.es) == 0  # the Knowledge Base was initialized with no edges

    def test_add_component_thread(self):
        """
        We can add a single ComponentThread object.

        Returns:
            None
        """
        so = SelfOrganize()
        so.add_component_threads(CLIP)
        assert len(so.graph.vs) == 1  # the Self-Organize Knowledge Base now has 1 vertex
        assert len(so.graph.es) == 0  # the Self-Organize Knowledge Base still has 0 edges

    def test_add_component_threads(self):
        """
        We can add an iterable collection of ComponentThread objects.

        Returns:
            None
        """
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_component_threads(functions)
        assert len(so.graph.vs) == 4  # the Self-Organize Knowledge Base now has 4 vertices
        assert len(so.graph.es) == 0  # the Self-Organize Knowledge Base still has 0 edges

    def test_add_edge(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_component_threads(functions)
        edges = (CLIP, WM, 1)  # CLIP produces antecedents & WM expects 2nd argument to be antecedents
        so.link_component_threads(edges)
        assert len(so.graph.vs) == 4  # the Self-Organize Knowledge Base now has 4 vertices
        assert len(so.graph.es) == 1  # the Self-Organize Knowledge Base has 1 edge

    def test_add_edges(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_component_threads(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_component_threads(edges)
        assert len(so.graph.vs) == 4  # the Self-Organize Knowledge Base now has 4 vertices
        assert len(so.graph.es) == 2  # the Self-Organize Knowledge Base has 2 edges

        edges = [
            (ECM, WM, 0),
        ]
        so.link_component_threads(edges)

    def test_add_input_data(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_component_threads(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_component_threads(edges)
        so.add_data(torch.rand(10, 5), name='input')
        assert len(so.graph.vs) == 5  # the Self-Organize Knowledge Base now has 5 vertices (incl. data vertex)
        assert len(so.graph.es) == 2  # the Self-Organize Knowledge Base has 2 edges

        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
        ]
        so.link_component_threads(edges)
        assert len(so.graph.es) == 4  # the Self-Organize Knowledge Base has 4 edges

    def test_start_threads(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_component_threads(functions)
        so.add_data(torch.rand(10, 5), name='input')
        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_component_threads(edges)
        so.start_threads(functions)
