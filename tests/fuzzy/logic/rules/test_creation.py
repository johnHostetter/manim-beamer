import torch
import unittest
import numpy as np

from tests.fuzzy.logic.rules.wang_mendel import rule_creation as old_WM_method
from soft.fuzzy.logic.rules.creation import wang_mendel_method as new_WM_method
from soft.fuzzy.online.unsupervised.cluster.ecm import ECM as newECM
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP as newCLIP
# local implementations of CLIP that we know work but are written in Numpy
from tests.fuzzy.online.unsupervised.granulation.clip import CLIP as oldCLIP


class TestWangMendelMethod(unittest.TestCase):
    def test_consistency(self):
        train_X = np.load('clip_input.npy')
        train_X_mins = train_X.min(axis=0)
        train_X_maxes = train_X.max(axis=0)

        # oldCLIP_terms = oldCLIP(train_X, train_X_mins, train_X_maxes)
        newCLIP_terms = newCLIP(torch.tensor(train_X), train_X_mins, train_X_maxes)

        oldCLIP_terms = []
        for var_idx, variable in enumerate(newCLIP_terms):
            oldCLIP_terms.append([])
            for idx, center in enumerate(variable.centers.detach().numpy()):
                value = {'center': center, 'sigma': variable.sigmas[idx].item(), 'support': 1}
                oldCLIP_terms[var_idx].append(value)

        Dthr = 0.4
        train_X = np.load('ecm_input.npy')
        new_clusters = newECM(train_X, Dthr=Dthr)
        reduced_X = new_clusters.centers
        old_antecedents, old_rules, _ = old_WM_method(reduced_X.detach().numpy(), oldCLIP_terms, [], [],
                                                      consistency_check=False)

        new_antecedents, new_rules = new_WM_method(reduced_X, newCLIP_terms)

        old_rules_matrix = np.array([rule['A'] for rule in old_rules])
        new_rules_matrix = np.array([rule.antecedents for rule in new_rules])

        assert old_rules_matrix.shape == new_rules_matrix.shape
        assert (old_rules_matrix == new_rules_matrix).all()
