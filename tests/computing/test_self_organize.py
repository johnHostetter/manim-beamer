import torch
import unittest

from utils.reproducibility import set_rng
from soft.computing.organize import stack_granules
from soft.computing.design import SelfOrganize, expert_design
from soft.computing.blueprints import clip_ecm_wm, clip_ftarm
from soft.fuzzy.relation.tnorm import AlgebraicProduct, Minimum
from soft.computing.wrappers import fetch_fuzzy_set_centers, FTARM

"""
The following algorithms are eligible for self-organizing neuro-fuzzy networks.
"""

from soft.fuzzy.online.unsupervised.cluster.ecm import ECM
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP
from soft.fuzzy.logic.rules.creation import wang_mendel_method as WM
from soft.fuzzy.offline.unsupervised.cluster.empirical import Empirical as EFS


set_rng(0)


def test_kwargs(so, testing_function):
    # find the vertex for this function & its predecessors
    target_vertex = so.graph.vs.find(function_eq=testing_function)
    predecessors_indices = so.graph.predecessors(target_vertex)
    predecessors_vertices = so.graph.vs[predecessors_indices]
    return so._SelfOrganize__get_kwargs(testing_function, predecessors_vertices, target_vertex)


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
        self.data = torch.rand(10, 5)

    def test_empty_self_organize_kb(self):
        so = SelfOrganize()
        assert len(so.graph.vs) == 0  # the Self-Organize Knowledge Base was initialized with no vertices
        assert len(so.graph.es) == 0  # the Self-Organize Knowledge Base was initialized with no edges

    def test_add_component_thread(self):
        """
        We can add a single ComponentThread object.

        Returns:
            None
        """
        so = SelfOrganize()
        so.add_functions(CLIP)
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
        so.add_functions(functions)
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
        so.add_functions(functions)
        edges = (CLIP, WM, 1)  # CLIP produces antecedents & WM expects 2nd argument to be antecedents
        so.link_functions(edges)
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
        so.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_functions(edges)
        assert len(so.graph.vs) == 4  # the Self-Organize Knowledge Base now has 4 vertices
        assert len(so.graph.es) == 2  # the Self-Organize Knowledge Base has 2 edges

        edges = [
            (ECM, WM, 0),
        ]
        so.link_functions(edges)

    def test_add_input_data(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_functions(edges)
        so.add_data(self.data, name='input')
        assert len(so.graph.vs) == 5  # the Self-Organize Knowledge Base now has 5 vertices (incl. data vertex)
        assert len(so.graph.es) == 2  # the Self-Organize Knowledge Base has 2 edges

        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
        ]
        so.link_functions(edges)
        assert len(so.graph.es) == 4  # the Self-Organize Knowledge Base has 4 edges

    def test_get_kwargs(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        so.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        so.link_functions(edges)
        so.add_data(self.data, name='input')
        assert len(so.graph.vs) == 5  # the Self-Organize Knowledge Base now has 5 vertices (incl. data vertex)
        assert len(so.graph.es) == 2  # the Self-Organize Knowledge Base has 2 edges

        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
        ]
        so.link_functions(edges)
        assert len(so.graph.es) == 4  # the Self-Organize Knowledge Base has 4 edges

        # find the vertex for this function & its predecessors
        target_vertex = so.graph.vs.find(function_eq=WM)
        predecessors_indices = so.graph.predecessors(target_vertex)
        predecessors_vertices = so.graph.vs[predecessors_indices]

        expected_kwargs = {'antecedents': None, 'input_data': None}
        assert so._SelfOrganize__get_kwargs(WM, predecessors_vertices, target_vertex) == expected_kwargs

    def test_start(self):
        so = SelfOrganize()
        functions = [
            CLIP,
            ECM,
            EFS,
            WM,
            stack_granules,
            fetch_fuzzy_set_centers,
            expert_design
        ]
        so.add_functions(functions)
        so.add_data(self.data, name='input')
        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
            (ECM, fetch_fuzzy_set_centers, 0),
            (fetch_fuzzy_set_centers, WM, 0),
            (CLIP, WM, 1),
            (CLIP, expert_design, 0),
            (WM, expert_design, 1)
        ]
        so.link_functions(edges)
        kb = so.start(functions)

        # --- test that information flowed properly from input data to CLIP ---
        testing_function = CLIP
        actual_kwargs = test_kwargs(so, testing_function)
        expected_kwargs = {'data': self.data}
        assert torch.isclose(actual_kwargs['data'], expected_kwargs['data']).all()

        # --- test that information flowed properly from input data to ECM ---
        testing_function = ECM
        actual_kwargs = test_kwargs(so, testing_function)
        expected_kwargs = {'data': self.data}
        assert torch.isclose(actual_kwargs['data'], expected_kwargs['data']).all()

        # --- test that information flowed properly from ECM to fetch_fuzzy_set_centers ---
        testing_function = fetch_fuzzy_set_centers
        ecm_output = so.graph.vs.find(function_eq=ECM)['output']
        actual_kwargs = test_kwargs(so, testing_function)
        expected_kwargs = {'fuzzy_sets': ecm_output}
        assert torch.isclose(actual_kwargs['fuzzy_sets'].centers, expected_kwargs['fuzzy_sets'].centers).all()
        assert torch.isclose(actual_kwargs['fuzzy_sets'].widths, expected_kwargs['fuzzy_sets'].widths).all()

        # --- test that information flowed properly from CLIP and fetch_fuzzy_set_centers to Wang-Mendel ---
        testing_function = WM
        antecedents = so.graph.vs.find(function_eq=CLIP)['output']
        input_data = so.graph.vs.find(function_eq=fetch_fuzzy_set_centers)['output']
        actual_kwargs = test_kwargs(so, testing_function)
        expected_kwargs = {'antecedents': antecedents, 'input_data': input_data}
        assert actual_kwargs['antecedents'] == expected_kwargs['antecedents']
        assert torch.isclose(actual_kwargs['input_data'], expected_kwargs['input_data']).all()

        # --- test that information flowed properly from CLIP and Wang-Mendel to expert_design ---
        testing_function = expert_design
        antecedents = so.graph.vs.find(function_eq=CLIP)['output']
        rules = so.graph.vs.find(function_eq=WM)['output']
        actual_kwargs = test_kwargs(so, testing_function)
        expected_kwargs = {'antecedents': antecedents, 'rules': rules}
        assert actual_kwargs['antecedents'] == expected_kwargs['antecedents']
        assert actual_kwargs['rules'] == expected_kwargs['rules']

        # check that the fuzzy logic rules are added

        assert len(kb.graph.vs.select(relation_eq=AlgebraicProduct)) == 9  # there should be 9 rules

        # checking that this query returns the same as the above; they are equivalent
        kb = so.graph.vs.find(function_eq=expert_design)['output']
        assert len(kb.graph.vs.select(relation_eq=AlgebraicProduct)) == 9  # there should be 9 rules

    def test_blueprint_clip_ecm_wm(self):
        so = clip_ecm_wm(self.data)
        kb = so.start()
        assert len(kb.graph.vs.select(relation_eq=AlgebraicProduct)) == 10  # there should be 10 rules

        # checking that this query returns the same as the above; they are equivalent
        kb = so.graph.vs.find(function_eq=expert_design)['output']
        assert len(kb.graph.vs.select(relation_eq=AlgebraicProduct)) == 10  # there should be 10 rules

    def test_blueprint_clip_ftarm(self):
        so = clip_ftarm(self.data)
        kb = so.start()
        assert len(kb.graph.vs.select(relation_eq=Minimum)) == 107  # there should be 107 rules

        # checking that this query returns the same as the above; they are equivalent
        kb = so.graph.vs.find(function_eq=FTARM)['output']
        assert len(kb.graph.vs.select(relation_eq=Minimum)) == 107  # there should be 107 rules
