import torch
import unittest
import numpy as np

from soft.fuzzy.sets import Gaussian
from utils.reproducibility import set_rng
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.information.granulation import GranulesMap
from soft.fuzzy.logic.inference.engines import ProductInference
from soft.fuzzy.logic.control.tsk import make_links_from_antecedents_to_rules

set_rng(0)


class FuzzyRuleBase:
    def __init__(self, rules):
        self.rules = rules
        self.antecedents_matrix_form = np.array([rule.antecedents for rule in self.rules])
        for col_idx in reversed(range(self.antecedents_matrix_form.shape[1])):
            indices = self.antecedents_matrix_form[:, col_idx].argsort()
            self.rules = [self.rules[idx] for idx in indices]
            self.antecedents_matrix_form = self.antecedents_matrix_form[indices]

    def num_of_rules(self):
        return self.antecedents_matrix_form.shape[0]

    def num_of_input_variables(self):
        return self.antecedents_matrix_form.shape[1]


def make_sparse_links(input_granulation, rules):
    frb = FuzzyRuleBase(rules)
    links = np.zeros((input_granulation.num_of_granules.sum(), len(rules)))
    col = np.zeros(input_granulation.num_of_granules.sum())  # where we are starting from
    row = np.zeros(len(rules))  # where we are going to
    for rule_idx, rule in enumerate(rules):
        for input_variable_idx, term_idx in enumerate(rule.antecedents):
            if isinstance(rule, Rule):
                actual_term_idx = input_granulation.iloc(input_variable_idx, term_idx)
            elif isinstance(rule, tuple):
                actual_term_idx = term_idx  # in this case, tuples have the correct index (FTARM algorithm)
            links[actual_term_idx, rule_idx] = 1

    # num_connections = 10
    col = torch.arange(frb.num_of_input_variables()).repeat_interleave(frb.num_of_rules()).view(1, -1).long()
    # row = torch.randint(low=0, high=2000, size=(784 * num_connections,)).view(1, -1).long()
    row = torch.tensor(frb.antecedents_matrix_form).transpose(1, 0).flatten().view(1, -1).long()
    connections = torch.cat((row, col), dim=0)
    return connections


class TestProductInference(unittest.TestCase):
    def test_output(self):
        in_features, out_features = 2, 2
        antecedents = [
            Gaussian(3, centers=torch.tensor([-1., 0., 1.]), widths=torch.tensor([1., 1., 1.])),
            Gaussian(3, centers=torch.tensor([-1., 0., 1.]), widths=torch.tensor([1., 1., 1.]))]
        rules = [
            Rule(antecedents=[0, 0], consequents=[0]),
            Rule(antecedents=[0, 1], consequents=[0]),
            Rule(antecedents=[1, 0], consequents=[0]),
            Rule(antecedents=[1, 1], consequents=[0]),
            Rule(antecedents=[2, 2], consequents=[0]),
            Rule(antecedents=[1, 2], consequents=[0]),
        ]
        input_granulation = GranulesMap(in_features=in_features, granules_params=antecedents,
                                        membership_function=Gaussian, trainable=False)
        links = make_links_from_antecedents_to_rules(input_granulation, rules)
        links = (torch.tensor(links).float())
        num_of_consequent_terms = len(rules)
        consequences = torch.nn.parameter.Parameter(torch.zeros(num_of_consequent_terms, out_features))

        consequences.requires_grad = True
        pi = ProductInference(in_features=in_features, out_features=out_features, consequences=consequences,
                              links=links)
        X = torch.randn((4, 2))
        antecedents_memberships = input_granulation(X)
        try:
            terms_to_rules = antecedents_memberships[:, :, None] \
                             * torch.tensor(pi.links_between_antecedents_and_rules).float()
        except IndexError:
            terms_to_rules = antecedents_memberships[:, None] * torch.tensor(
                pi.links_between_antecedents_and_rules)

        with torch.no_grad():
            terms_to_rules[terms_to_rules == 0] = 1.0  # ignore zeroes, this is from the weights between terms and rules
        # import sparselinear as sl
        # connections = make_sparse_links(input_granulation, rules)
        frb = FuzzyRuleBase(rules)
        num_of_observations = X.shape[0]
        num_of_rules = frb.num_of_rules()
        num_of_variables = X.shape[1]
        num_of_terms = 3
        actual_output = (antecedents_memberships[:, None] * links.transpose(0, 1))\
            .reshape(num_of_observations, num_of_rules, num_of_variables, num_of_terms).sum(-1)\
            .transpose(1, 2).prod(dim=1)
        # the shape of terms_to_rules is (num of observations, num of ALL terms, num of rules)
        rules_applicability = terms_to_rules.prod(dim=1).float()
        expected_output = torch.tensor([[0.00475898, 0.06913665, 0.02898348, 0.4210613, 0.68233776,
                                         0.8278493],
                                        [0.6473842, 0.78544694, 0.41755858, 0.5066081, 0.00726113,
                                         0.08318364],
                                        [0.21015427, 0.8221435, 0.17322151, 0.67765903, 0.04002269,
                                         0.3587828],
                                        [0.01187785, 0.291746, 0.00582015, 0.14295565, 0.03151279,
                                         0.4752033]])
        assert torch.isclose(actual_output, expected_output).all()
