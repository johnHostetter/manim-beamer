import torch
import unittest

from soft.fuzzy.sets import Gaussian
from utils.reproducibility import set_rng
from soft.fuzzy.logic.rules.knowledge import Rule
from soft.fuzzy.information.granulation import GranulesMap
from soft.fuzzy.logic.inference.engines import ProductInference
from soft.fuzzy.logic.control.tsk import make_links_from_antecedents_to_rules


set_rng(0)


class TestProductInference(unittest.TestCase):
    def test_output(self):
        in_features, out_features = 2, 2
        antecedents = [
            Gaussian(3, centers=torch.tensor([-1, 0., 1.]), widths=torch.tensor([1., 1., 1.])),
            Gaussian(3, centers=torch.tensor([-1., 0., 1.]), widths=torch.tensor([1., 1., 1.]))]
        rules = [
            Rule(antecedents=[0, 0], consequents=[0]),
            Rule(antecedents=[0, 1], consequents=[0]),
            Rule(antecedents=[1, 0], consequents=[0]),
            Rule(antecedents=[1, 1], consequents=[0]),
            Rule(antecedents=[1, 2], consequents=[0]),
            Rule(antecedents=[1, 2], consequents=[0]),
        ]
        input_granulation = GranulesMap(in_features=in_features, granules_params=antecedents,
                                        membership_function=Gaussian, trainable=False)
        links = make_links_from_antecedents_to_rules(input_granulation, rules)
        num_of_consequent_terms = len(rules)
        consequences = torch.nn.parameter.Parameter(torch.zeros(num_of_consequent_terms, out_features))
        consequences.requires_grad = True
        pi = ProductInference(in_features=in_features, out_features=out_features, consequences=consequences,
                              links=links)
        X = torch.randn((4, 2))
        antecedents_memberships = input_granulation(X)
        actual_output = pi.calc_rules_applicability(antecedents_memberships)
        expected_output = torch.tensor([[9.52992122e-04, 1.44050468e-03, 5.64775779e-02, 8.53692423e-02,
                                         1.74637646e-02, 1.74637646e-02],
                                        [2.12899314e-02, 1.80385602e-01, 7.41303946e-04, 6.28092951e-03,
                                         7.20215626e-03, 7.20215626e-03],
                                        [8.47027257e-01, 1.40406535e-01, 2.63140523e-01, 4.36191975e-02,
                                         9.78540533e-04, 9.78540533e-04],
                                        [4.75897548e-03, 6.91366485e-02, 2.89834770e-02, 4.21061312e-01,
                                         8.27849315e-01, 8.27849315e-01]])
        assert torch.isclose(actual_output, expected_output).all()
