import torch
import pickle
import unittest

from utils.reproducibility import set_rng
from soft.fuzzy.logic.inference.engines import ProductInference
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP
from soft.fuzzy.logic.control.tsk import make_links_from_antecedents_to_rules


class TestCriteria(unittest.TestCase):
    def test_noise_criterion(self):
        set_rng(0)
        num_of_observations, num_of_inputs, num_of_outputs = 20, 5, 1
        X = torch.randint(high=5, size=(num_of_observations, num_of_inputs))
        Y = torch.randint(high=10, size=(num_of_observations, num_of_outputs)).float()
        output_terms = CLIP(Y, Y.detach().numpy().min(axis=0), Y.detach().numpy().max(axis=0))
        infile = open('ftarm_demo_frb', 'rb')
        frb = pickle.load(infile)  # there should be 6 rules
        infile.close()
        infile = open('ftarm_demo_granulation', 'rb')
        granulation = pickle.load(infile)
        infile.close()
        consequences = [rule.consequents for rule in
                        frb.rules]  # dummy variable; doesn't matter for our usage
        links = make_links_from_antecedents_to_rules(granulation, frb.rules)
        pi = ProductInference(in_features=granulation.granules.centers.shape[0], out_features=1,
                              consequences=torch.tensor(consequences), links=links)

        antecedents_memberships = granulation(X.float())
        # rules_memberships.shape = (num. of observations, num. of rules)
        rules_memberships = pi.calc_rules_applicability(antecedents_memberships)
        # output_memberships.shape = (num. of observations, 1)
        output_memberships = output_terms[0](Y)

        # noise criterion formula, the original formula uses min inside the minimum, but this is incorrect
        # since data that strongly belongs to one or more clusters may be incorrectly labeled as noise if it has zero
        # membership to other clusters (either inside input or output space)
        NC = 0.1  # observations with values less than NC are considered noise
        values = torch.minimum(rules_memberships.max(dim=-1).values.flatten(),
                             output_memberships.max(dim=-1).values.flatten())
        actual_noise = values < NC
        expected_noise = torch.tensor(
            [False, False, False, False, False, False,  True, False, False, False,
             False, False, False, False, False, False, False, False, False, False]
        )
        assert (actual_noise == expected_noise).all()
