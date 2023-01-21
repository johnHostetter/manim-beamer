import os
import torch
import pathlib
import unittest
import numpy as np

from soft.fuzzy.online.unsupervised.cluster.ecm import ECM as newECM
from soft.fuzzy.online.unsupervised.granulation.clip import CLIP as newCLIP
from tests.fuzzy.logic.rules.wang_mendel import rule_creation as old_WM_method
from soft.fuzzy.logic.rules.creation import wang_mendel_method as new_WM_method


class TestWangMendelMethod(unittest.TestCase):
    def test_consistency(self):
        directory = pathlib.Path(__file__).parent.resolve()
        file_location = os.path.join(directory, 'clip_input.npy')
        train_X = np.load(file_location)

        # oldCLIP_terms = oldCLIP(train_X, train_X_mins, train_X_maxes)
        newCLIP_terms = newCLIP(torch.tensor(train_X))

        oldCLIP_terms = []
        for var_idx, variable in enumerate(newCLIP_terms):
            oldCLIP_terms.append([])
            for idx, center in enumerate(variable.centers.detach().numpy()):
                value = {'center': center, 'sigma': variable.sigmas[idx].item(), 'support': 1}
                oldCLIP_terms[var_idx].append(value)

        directory = pathlib.Path(__file__).parent.resolve()
        file_location = os.path.join(directory, 'ecm_input.npy')
        train_X = np.load(file_location)
        new_clusters = newECM(torch.tensor(train_X), config={'dthr': 0.4})
        reduced_X = new_clusters.centers
        old_antecedents, old_rules, _ = old_WM_method(reduced_X.detach().numpy(), oldCLIP_terms, [], [],
                                                      consistency_check=False)

        new_rules = new_WM_method(reduced_X, newCLIP_terms)

        old_rules_matrix = np.array([rule['A'] for rule in old_rules])
        new_rules_matrix = np.array([rule.antecedents for rule in new_rules])

        assert old_rules_matrix.shape == new_rules_matrix.shape
        assert (old_rules_matrix == new_rules_matrix).all()
