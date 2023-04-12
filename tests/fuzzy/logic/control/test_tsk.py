import unittest
import torch

from utils.reproducibility import set_rng
from soft.fuzzy.sets.continuous import Gaussian
from soft.computing.design import expert_design
from soft.fuzzy.logic.control.tsk import ZeroOrderTSK
from soft.fuzzy.relation.tnorm import AlgebraicProduct


set_rng(0)


class TestTSK(unittest.TestCase):
    """
    Test the zero-order TSK neuro-fuzzy network.
    """
    def test_gradient_1(self):
        """
        First test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()
        # first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(torch.tensor([[0, 1], [1, 2], [2, 3]]).float())
        actual_result = input_data.unsqueeze(dim=-1) - centers.T
        expected_result = torch.tensor([[[1.2000, 0.2000, -0.8000],
                                         [-0.8000, -1.8000, -2.8000]],
                                        [[1.1000, 0.1000, -0.9000],
                                         [-0.7000, -1.7000, -2.7000]],
                                        [[2.1000, 1.1000, 0.1000],
                                         [-0.9000, -1.9000, -2.9000]],
                                        [[2.7000, 1.7000, 0.7000],
                                         [-0.8500, -1.8500, -2.8500]],
                                        [[1.7000, 0.7000, -0.3000],
                                         [-0.7500, -1.7500, -2.7500]]])

        assert torch.isclose(actual_result, expected_result).all()

    def test_gradient_2(self):
        """
        Second test that the gradient of PyTorch is working as intended.

        Returns:
            None
        """
        value_1 = torch.nn.Parameter(torch.tensor([0, 1]).float())
        value_3 = 2 ** value_1
        assert value_3.grad_fn is not None

    def test_tsk(self):
        """
        Test the zero-order TSK neuro-fuzzy network.

        Returns:
            None
        """
        input_data = torch.tensor([
            [1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]
        ]).float()
        actual_y = torch.tensor([1.5, 0.6, 0.9, 0.7, 1.3]).float()

        antecedents = [
            Gaussian(4, centers=torch.tensor([1.2, 3.0, 5.0, 7.0]).float(),
                     widths=torch.tensor([0.1, 0.4, 0.6, 0.8]).float()),
            Gaussian(4, centers=torch.tensor([0.2, 0.6, 0.9, 1.2]).float(),
                     widths=torch.tensor([0.4, 0.4, 0.5, 0.45]).float())
        ]

        # check that antecedents were correctly created
        assert all(antecedents[0].centers == torch.tensor([1.2, 3.0, 5.0, 7.0]))
        assert all(antecedents[0].widths == torch.tensor([0.1, 0.4, 0.6, 0.8]))
        assert all(antecedents[1].centers == torch.tensor([0.2, 0.6, 0.9, 1.2]))
        assert all(antecedents[1].widths == torch.tensor([0.4, 0.4, 0.5, 0.45]))

        rules = {
            frozenset({(0, 0), (1, 0)}), frozenset({(0, 1), (1, 0)}),
            frozenset({(0, 1), (1, 1)}), frozenset({(1, 1), (1, 1)})
        }
        knowledge_base = expert_design(antecedents, rules, config={})

        rule_vertex = knowledge_base.graph.vs.find(relation_eq=AlgebraicProduct)
        assert rule_vertex['relation'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'id' in rule_vertex.attributes()  # it has a unique id

        rule_vertices = knowledge_base.graph.vs.select(relation_eq=AlgebraicProduct)
        assert len(rule_vertices) == 4  # there should be 4 fuzzy logic rules

        # there should be 2 rules that use (1, 1);
        # the last rule has been simplified (redundant mention of condition)
        assert knowledge_base[(1, 1)] == {
            AlgebraicProduct: [frozenset({(0, 1), (1, 1)}), frozenset({(1, 1)})]
        }

        # the rules we have added should exist how we expected them
        assert knowledge_base.edges(AlgebraicProduct) == rules

        knowledge_base.attributes(rule_vertex['name'])
        n_output = actual_y.ndim
        flc = ZeroOrderTSK(n_output, knowledge_base, input_trainable=True)
        predicted_y = flc(input_data)
        print(predicted_y)

        print(f"sigmas: {flc.input_granulation.sigmas}")
        print(f"log widths: {flc.input_granulation._log_widths}")
        print(f"centers: {flc.input_granulation.centers}")
        print(f"consequences: {flc.consequences}")
