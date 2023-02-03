import torch
import unittest

from utils.reproducibility import set_rng
from soft.computing.graph import KnowledgeBase
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.logic.control.tsk import ZeroOrderTSK
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.information.granulation import GranulesMap


set_rng(0)


class TestTSK(unittest.TestCase):
    def test_gradient_1(self):
        X = torch.tensor([[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]]).float()
        # first variable has fuzzy sets with centers 0, 1, 2 (the column)
        centers = torch.nn.Parameter(torch.tensor([[0, 1], [1, 2], [2, 3]]).float())
        actual_result = X.unsqueeze(dim=-1) - centers.T
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

        actual_result = -1.0 * (torch.pow(actual_result, 2))

        sigmas = torch.nn.Parameter(torch.tensor([[0.0867, 0.3339],
                                                  [0.0518, 0.8080],
                                                  [0.0578, 0.3440]]))

    def test_gradient_2(self):
        a = torch.nn.Parameter(torch.tensor([0, 1]).float())
        b = torch.nn.Parameter(torch.tensor([1, 2]).float())
        c = 2 ** a
        assert c.grad_fn is not None

    def test_tsk(self):
        x = torch.tensor([[1.2, 0.2], [1.1, 0.3], [2.1, 0.1], [2.7, 0.15], [1.7, 0.25]]).float()
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

        kb = KnowledgeBase(antecedents)
        kb.add(AlgebraicProduct, [((0, 0), (1, 0)), ((0, 1), (1, 0)), ((0, 1), (1, 1)), ((1, 1), (1, 1))])
        gm = GranulesMap(kb=kb)
        rule_vertex = kb.graph.vs.find(relation_eq=AlgebraicProduct)
        assert rule_vertex['relation'] == AlgebraicProduct  # it is the correct relation we wanted
        assert 'id' in rule_vertex.attributes()  # it has a unique id
        rule_vertices = kb.graph.vs.select(relation_eq=AlgebraicProduct)
        assert len(rule_vertices) == 4  # there should be 4 fuzzy logic rules
        # there should be 2 rules that use (1, 1); the last rule has been simplified (redundant mention of condition)
        assert kb.graph[(1, 1)] == {AlgebraicProduct: [frozenset({(0, 1), (1, 1)}), frozenset({(1, 1)})]}

        kb.graph.attributes(rule_vertex['id'])
        n_output = actual_y.ndim
        flc = ZeroOrderTSK(n_output, kb, input_trainable=True)
        predicted_y = flc(x)
        print(predicted_y)

        print('sigmas: {}'.format(flc.input_granulation.sigmas))
        print('log widths: {}'.format(flc.input_granulation._log_widths))
        print('centers: {}'.format(flc.input_granulation.centers))
        print('consequences: {}'.format(flc.consequences))
