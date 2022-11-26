import torch
import unittest
import numpy as np

from soft.fuzzy.logic.control.tsk import make_links_from_antecedents_to_rules
from soft.fuzzy.sets import Triangular
from soft.fuzzy.information.granulation import GranulesMap
from examples.fuzzy.temporal.association.ftarm.sample import make_example
from soft.fuzzy.temporal.association.ftarm import FTARM, TemporalInformationTable as TI


class TestFTARM(unittest.TestCase):
    def test_fuzzy_representation(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        expected_membership = torch.tensor([[1.0000, 0.0000, 0.0000, 0.0000, 2 / 3, 1 / 3, 0.0000, 0.0000, 0.0000,
                                             0.0000],
                                            [0.6000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                             0.0000],
                                            [0.0000, 0.0000, 0.0000, 0.0000, 2 / 3, 1 / 3, 0.0000, 0.0000, 0.0000,
                                             0.0000],
                                            [0.6000, 0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000,
                                             0.0000],
                                            [0.6000, 0.0000, 0.7500, 0.2500, 0.0000, 0.0000, 0.0000, 1.0000, 1.0000,
                                             0.0000]])
        assert (torch.isclose(mus, expected_membership)).all()

    def test_temporal_information_table(self):
        dataframe, terms = make_example()
        ti_table = TI(dataframe, terms)
        # temporal item A occurs in the first transaction, B occurs in the second, C occurs in the first, and so on
        assert (ti_table.first_transaction_indices == np.array([0, 1, 0, 3, 4])).all()
        # temporal items D and E come in the second time period, all others occur in the first time period
        assert (ti_table.starting_periods.values == np.array([[0, 0, 0, 1, 1]])).all()
        # ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA

    def test_step_2(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        # check that the variable location method is working properly (returns start and end indices)
        for variable_index in range(5):
            assert granulation.vloc(variable_index) == (2 * variable_index, 2 * variable_index + 1)
        expected_scalar_cardinality = torch.tensor(
            [2.8000, 0.0000, 1.7500, 0.2500, 4 / 3, 2 / 3, 0.0000, 2.0000, 1.0000,
             0.0000])
        assert torch.isclose(mus.sum(dim=0), expected_scalar_cardinality).all()

    def test_step_3(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        # check that the variable location method is working properly (returns start and end indices)
        for variable_index in range(5):
            assert granulation.vloc(variable_index) == (2 * variable_index, 2 * variable_index + 1)
        expected_scalar_cardinality = torch.tensor(
            [2.8000, 0.0000, 1.7500, 0.2500, 4 / 3, 2 / 3, 0.0000, 2.0000, 1.0000,
             0.0000])
        numerator = mus.sum(dim=0)
        assert torch.isclose(numerator, expected_scalar_cardinality).all()
        ti_table = TI(dataframe, terms)
        num_of_possible_transactions_per_temporal_item = [ti_table.size_of_transactions_per_time_granule.values[idx:]
                                                          .sum() for idx in ti_table.starting_periods.values[0]]
        denominator = granulation.linear_map(torch.tensor(num_of_possible_transactions_per_temporal_item).float())
        fuzzy_temporal_supports = numerator / denominator

    def test_step_4(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        # check that the variable location method is working properly (returns start and end indices)
        for variable_index in range(5):
            assert granulation.vloc(variable_index) == (2 * variable_index, 2 * variable_index + 1)
        numerator = mus.sum(dim=0)
        ti_table = TI(dataframe, terms)
        num_of_possible_transactions_per_temporal_item = [ti_table.size_of_transactions_per_time_granule.values[idx:]
                                                          .sum() for idx in ti_table.starting_periods.values[0]]
        denominator = granulation.linear_map(torch.tensor(num_of_possible_transactions_per_temporal_item).float())
        fuzzy_temporal_supports = numerator / denominator
        # start of step 4
        minimum_support = 0.3
        fuzzy_temporal_supports >= minimum_support  # L1 --> low A, low B, high D, and low E
        assert ((fuzzy_temporal_supports >= minimum_support) ==
                torch.tensor([True, False, True, False, False, False, False, True, True, False])).all()

        if torch.count_nonzero(fuzzy_temporal_supports >= minimum_support).item() > 0:  # step 5
            r = 1  # then we continue (with step 6), where r is the number of items in the current itemsets

        L1_indices = torch.where(fuzzy_temporal_supports >= minimum_support)[0]  # the indices for the L1 items

        from itertools import combinations

        # (low A, low B), (low A, high D), (low A, low E), (low B, high D), (low B, low E), (high D, low E)
        C2_indices = list(combinations(L1_indices.tolist(), r=r+1))

        from collections import namedtuple
        FakeRule = namedtuple('FakeRule', 'antecedents')
        fake_rules = [FakeRule(candidate_indices) for candidate_indices in C2_indices]
        links = make_links_from_antecedents_to_rules(granulation, fake_rules)

        (mus[:, :, None] * torch.tensor(links).float())
    # def test_step_1(self):
    #     dataframe, terms = make_example()
    #     ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
    #     actual_output = ftarm._FTARM__step_1()
    #     with open('data/step_1_output.npy', 'rb') as file:
    #         expected_output = np.load(file)
    #     assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
    #
    # def test_step_2(self):
    #     dataframe, terms = make_example()
    #     ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
    #     f = ftarm._FTARM__step_1()
    #     actual_output = ftarm._FTARM__step_2(f)
    #     with open('data/step_2_output.npy', 'rb') as file:
    #         expected_output = np.load(file)
    #     assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
    #
    # def test_step_3(self):
    #     dataframe, terms = make_example()
    #     ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
    #     f = ftarm._FTARM__step_1()
    #     count = ftarm._FTARM__step_2(f)
    #     actual_output = ftarm._FTARM__step_3(count)
    #     with open('data/step_3_output.npy', 'rb') as file:
    #         expected_output = np.load(file)
    #     assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
    #
    # def test_step_4(self):
    #     dataframe, terms = make_example()
    #     ftarm = FTARM(dataframe, terms, ALPHA=0.3, LAMBDA=0.6)  # using default values for ALPHA and LAMBDA
    #     f = ftarm._FTARM__step_1()
    #     count = ftarm._FTARM__step_2(f)
    #     tFuzzySupport = ftarm._FTARM__step_3(count)
    #     actual_output = ftarm._FTARM__step_4(tFuzzySupport)
    #     with open('data/step_4_output.npy', 'rb') as file:
    #         expected_output = np.load(file)
    #     assert np.isclose(actual_output, expected_output, rtol=1e-8, equal_nan=True).all()
