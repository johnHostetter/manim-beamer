import os
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.organize import stack_granules
from soft.computing.knowledge import KnowledgeBase
from soft.computing.design import SelfOrganize, expert_design
from soft.fuzzy.relation.tnorm import AlgebraicProduct, Minimum
from soft.computing.wrappers import fetch_fuzzy_set_centers, FTARM
from soft.computing.blueprints import clip_ecm_wm, clip_ftarm, clip_frequent_discernible
# the following algorithms are eligible for self-organizing neuro-fuzzy networks
from soft.fuzzy.online.unsupervised.cluster.ecm import ECM
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP
from soft.fuzzy.offline.unsupervised.cluster.empirical import Empirical as EFS
from soft.fuzzy.logic.rules.creation import wang_mendel_method as WM, frequent_discernible


def test_kwargs(self_organize, testing_function):
    """
    Finds the vertex for the given function and its predecessors, then returns the
    keyword arguments' values given to the function (testing_function).

    Args:
        self_organize: A SelfOrganize object.
        testing_function: A callable function that is within SelfOrganize's vertices.

    Returns:
        The keyword arguments, and their values, that were given to the function (testing_function).
    """
    target_vertex = self_organize.graph.vs.find(function_eq=testing_function)
    predecessors_indices = self_organize.graph.predecessors(target_vertex)
    predecessors_vertices = self_organize.graph.vs[predecessors_indices]
    return self_organize._SelfOrganize__get_kwargs(
        testing_function, predecessors_vertices, target_vertex)


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
        self.data = torch.load('small_data.pt')
        self.configuration = {}

    def test_empty_self_organize_kb(self):
        """
        Test that if we create a SelfOrganize object and do nothing,
        that nothing is stored inside already.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        num_of_vertices, num_of_edges = 0, 0
        assert len(self_organize.graph.vs) == num_of_vertices
        assert len(self_organize.graph.es) == num_of_edges

    def test_add_component_thread(self):
        """
        We can add a single ComponentThread object.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        self_organize.add_functions(CLIP)
        num_of_vertices, num_of_edges = 1, 0
        assert len(self_organize.graph.vs) == num_of_vertices
        assert len(self_organize.graph.es) == num_of_edges

    def test_add_component_threads(self):
        """
        We can add an iterable collection of ComponentThread objects.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        self_organize.add_functions(functions)
        edges = []
        num_of_vertices, num_of_edges = len(functions), len(edges)
        assert len(self_organize.graph.vs) == num_of_vertices
        assert len(self_organize.graph.es) == num_of_edges

    def test_add_edge(self):
        """
        Test that adding an edge works as intended.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        self_organize.add_functions(functions)
        edges = (CLIP, WM, 1)  # CLIP produces antecedents & WM expects 2nd arg. to be antecedents
        self_organize.link_functions(edges)
        num_of_vertices, num_of_edges = len(functions), len([edges])
        assert len(self_organize.graph.vs) == num_of_vertices
        assert len(self_organize.graph.es) == num_of_edges

    def test_add_edges(self):
        """
        Test that adding several edges works as intended.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        self_organize.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        self_organize.link_functions(edges)
        num_of_vertices, num_of_edges = len(functions), len(edges)
        assert len(self_organize.graph.vs) == num_of_vertices
        assert len(self_organize.graph.es) == num_of_edges

        edges = [
            (ECM, WM, 0),
        ]
        self_organize.link_functions(edges)

    def test_add_input_data(self):
        """
        Test adding a special vertex to store the input data. The special vertex's value is passed
        as an argument to functions that rely upon input data.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        self_organize.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        self_organize.link_functions(edges)
        self_organize.add_data(self.data, name='input')
        num_of_vertices, num_of_edges = len(functions) + 1, len(edges)
        assert len(self_organize.graph.vs) == num_of_vertices  # (incl. data vertex)
        assert len(self_organize.graph.es) == num_of_edges

        more_edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
        ]
        self_organize.link_functions(more_edges)
        num_of_edges = len(edges) + len(more_edges)
        assert len(self_organize.graph.es) == num_of_edges

    def test_get_kwargs(self):
        """
        Test that the keyword arguments are as expected in SelfOrganize.

        Returns:
            None
        """
        set_rng(0)
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM
        ]
        self_organize.add_functions(functions)
        edges = [
            (ECM, WM, 0),
            (CLIP, WM, 1),
        ]
        self_organize.link_functions(edges)
        self_organize.add_data(self.data, name='input')
        num_of_vertices, num_of_edges = len(functions) + 1, len(edges)
        assert len(self_organize.graph.vs) == num_of_vertices  # (incl. data vertex)
        assert len(self_organize.graph.es) == num_of_edges

        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
        ]
        self_organize.link_functions(edges)
        assert len(self_organize.graph.es) == 4  # the Self-Organize Knowledge Base has 4 edges

        # find the vertex for this function & its predecessors
        target_vertex = self_organize.graph.vs.find(function_eq=WM)
        predecessors_indices = self_organize.graph.predecessors(target_vertex)
        predecessors_vertices = self_organize.graph.vs[predecessors_indices]

        expected_kwargs = {'antecedents': None, 'input_data': None}
        assert self_organize._SelfOrganize__get_kwargs(
            WM, predecessors_vertices, target_vertex) == expected_kwargs

    def test_start(self):
        """
        Test a verbose definition of a self-organizing process (i.e., no shortcut method call).

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        number_of_rules = 10
        self_organize = SelfOrganize(config=self.configuration)
        functions = [
            CLIP,
            ECM,
            EFS,
            WM,
            stack_granules,
            fetch_fuzzy_set_centers,
            expert_design
        ]
        self_organize.add_functions(functions)
        self_organize.add_data(self.data, name='input')
        self_organize.add_data(self.configuration, name='config')
        edges = [
            ('input', ECM, 0),
            ('input', CLIP, 0),
            (ECM, fetch_fuzzy_set_centers, 0),
            (fetch_fuzzy_set_centers, WM, 0),
            (CLIP, WM, 1),
            (CLIP, expert_design, 0),
            (WM, expert_design, 1),
            ('config', expert_design, 2)
        ]
        self_organize.link_functions(edges)
        knowledge_base = self_organize.start(functions)

        # --- test info flowed properly from input data to CLIP ---
        testing_function = CLIP
        actual_kwargs = test_kwargs(self_organize, testing_function)
        expected_kwargs = {'data': self.data}
        assert torch.isclose(actual_kwargs['data'], expected_kwargs['data']).all()

        # --- test info flowed properly from input data to ECM ---
        testing_function = ECM
        actual_kwargs = test_kwargs(self_organize, testing_function)
        expected_kwargs = {'data': self.data}
        assert torch.isclose(actual_kwargs['data'], expected_kwargs['data']).all()

        # --- test info flowed properly from ECM to fetch_fuzzy_set_centers ---
        testing_function = fetch_fuzzy_set_centers
        ecm_output = self_organize.graph.vs.find(function_eq=ECM)['output']
        actual_kwargs = test_kwargs(self_organize, testing_function)
        expected_kwargs = {'fuzzy_sets': ecm_output}
        assert torch.isclose(
            actual_kwargs['fuzzy_sets'].centers, expected_kwargs['fuzzy_sets'].centers).all()
        assert torch.isclose(
            actual_kwargs['fuzzy_sets'].widths, expected_kwargs['fuzzy_sets'].widths).all()

        # --- test info flowed properly from CLIP and fetch_fuzzy_set_centers to Wang-Mendel ---
        testing_function = WM
        antecedents = self_organize.graph.vs.find(function_eq=CLIP)['output']
        input_data = self_organize.graph.vs.find(function_eq=fetch_fuzzy_set_centers)['output']
        actual_kwargs = test_kwargs(self_organize, testing_function)
        expected_kwargs = {'antecedents': antecedents, 'input_data': input_data}
        assert actual_kwargs['antecedents'] == expected_kwargs['antecedents']
        assert torch.isclose(actual_kwargs['input_data'], expected_kwargs['input_data']).all()

        # --- test info flowed properly from CLIP and Wang-Mendel to expert_design ---
        testing_function = expert_design
        antecedents = self_organize.graph.vs.find(function_eq=CLIP)['output']
        rules = self_organize.graph.vs.find(function_eq=WM)['output']
        actual_kwargs = test_kwargs(self_organize, testing_function)
        expected_kwargs = {'antecedents': antecedents, 'rules': rules}
        assert actual_kwargs['antecedents'] == expected_kwargs['antecedents']
        assert actual_kwargs['rules'] == expected_kwargs['rules']

        # check that the fuzzy logic rules are added

        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = self_organize.graph.vs.find(function_eq=expert_design)['output']
        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        return knowledge_base

    def test_blueprint_clip_ecm_wm(self):
        """
        Test the self-organizing process with CLIP, followed by ECM, and then generate fuzzy logic
        rules with the Wang-Mendel method.

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        number_of_rules = 10
        self_organize = clip_ecm_wm(self.data, config={})
        knowledge_base = self_organize.start()
        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = self_organize.graph.vs.find(function_eq=expert_design)['output']
        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        return knowledge_base

    def test_blueprint_clip_ftarm(self):
        """
        Test the self-organizing process with CLIP followed by the fuzzy temporal association
        rule mining method.

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        number_of_rules = 142
        self_organize = clip_ftarm(self.data, config={'minimum_support': 0.3,
                                                      'minimum_confidence': 0.8})
        knowledge_base = self_organize.start()
        assert len(knowledge_base.graph.vs.select(relation_eq=Minimum)) == number_of_rules

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = self_organize.graph.vs.find(function_eq=FTARM)['output']
        assert len(knowledge_base.graph.vs.select(relation_eq=Minimum)) == number_of_rules

        return knowledge_base

    def test_blueprint_clip_frequent_discernible(self):
        """
        Test the self-organizing process with CLIP followed by the frequent discernible method.

        Returns:
            KnowledgeBase
        """
        set_rng(0)
        number_of_rules = 8
        big_train_data = torch.load('big_train_data.pt')
        big_val_data = torch.load('big_val_data.pt')
        config = {'lr': 1e-4, 'batch_size': 128, 'latent_space_dim': 2, 'max_epochs': 10}
        self_organize = clip_frequent_discernible(big_train_data, big_val_data, config)
        knowledge_base = self_organize.start()
        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        # checking that this query returns the same as the above; they are equivalent
        knowledge_base = self_organize.graph.vs.find(function_eq=frequent_discernible)['output']
        assert len(knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)) == number_of_rules

        return knowledge_base

    def test_save_load_knowledge_base(self):
        """
        Test that when we save and load the KnowledgeBase object,
        that we retrieve the original KnowledgeBase.

        Returns:
            None
        """
        set_rng(0)
        blueprints = [  # selected two methods
            self.test_blueprint_clip_ecm_wm,
            self.test_blueprint_clip_ftarm,
        ]
        path_to_this_script = Path(os.path.dirname(os.path.abspath(__file__)))
        file_path = path_to_this_script / "models"
        for blueprint in blueprints:
            knowledge_base = blueprint()
            file_name = knowledge_base.save(file_path)
            loaded_knowledge_base = KnowledgeBase.load(file_name)

            assert knowledge_base.config == loaded_knowledge_base.config
            assert knowledge_base._Core__index == loaded_knowledge_base._Core__index
            assert knowledge_base.attribute_table == loaded_knowledge_base.attribute_table
            assert (np.array([
                granule_vertex.index for granule_vertex in loaded_knowledge_base.granules]) == ([
                granule_vertex.index for granule_vertex in loaded_knowledge_base.granules])
                    ).all()

            for vertex, loaded_vertex in zip(
                    knowledge_base.graph.vs, loaded_knowledge_base.graph.vs):
                if isinstance(vertex['name'], Gaussian) and \
                        str(loaded_vertex['name'].__class__) == str(vertex['name'].__class__):
                    # loaded vertex's class is different
                    assert (torch.isclose(
                        vertex['name'].centers, loaded_vertex['name'].centers
                    ).all() and torch.isclose(
                        vertex['name'].widths, loaded_vertex['name'].widths
                    ).all()).item()
                else:
                    if not vertex.attributes() == loaded_vertex.attributes():
                        # Python classes are 'different' after reload
                        for attribute in vertex.attributes().keys():
                            assert str(vertex[attribute]) == str(loaded_vertex[attribute])
                    else:
                        assert vertex.attributes() == loaded_vertex.attributes()

            for edge, loaded_edge in zip(knowledge_base.graph.es, loaded_knowledge_base.graph.es):
                assert edge.attributes() == loaded_edge.attributes()

        shutil.rmtree(file_path)  # clean up; delete the model files
