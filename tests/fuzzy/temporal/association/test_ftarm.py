import torch
import unittest
import numpy as np

from soft.fuzzy.sets import Triangular
from soft.fuzzy.logic.rules.knowledge import Rule
from soft.fuzzy.information.granulation import GranulesMap
from soft.fuzzy.logic.inference.engines import MinimumInference
from examples.fuzzy.temporal.association.ftarm.sample import make_example
from soft.fuzzy.logic.control.tsk import make_links_from_antecedents_to_rules
from soft.fuzzy.temporal.association.ftarm import FTARM, TemporalInformationTable as TI


class TestFTARM(unittest.TestCase):
    def test_fuzzy_representation(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        expected_membership = torch.tensor([[1.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
        assert (torch.isclose(mus.reshape(5, 10), expected_membership)).all()

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
        expected_scalar_cardinality = torch.tensor([2.5, 0.0, 1.75, 0.25, 4 / 3, 2 / 3, 0.0, 2.0, 1.0, 0.0])
        assert torch.isclose(mus.sum(dim=0).flatten(), expected_scalar_cardinality).all()

    def test_step_3(self):
        dataframe, terms = make_example()
        granulation = GranulesMap(len(terms.keys()), list(terms.values()), Triangular)
        mus = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        # check that the variable location method is working properly (returns start and end indices)
        for variable_index in range(5):
            assert granulation.vloc(variable_index) == (2 * variable_index, 2 * variable_index + 1)
        expected_scalar_cardinality = torch.tensor([2.5, 0.0, 1.75, 0.25, 4 / 3, 2 / 3, 0.0, 2.0, 1.0, 0.0])
        numerator = mus.sum(dim=0)
        assert torch.isclose(numerator.flatten(), expected_scalar_cardinality).all()
        ti_table = TI(dataframe, terms)
        num_of_possible_transactions_per_temporal_item = [ti_table.size_of_transactions_per_time_granule.values[idx:]
                                                          .sum() for idx in ti_table.starting_periods.values[0]]
        denominator = torch.tensor(num_of_possible_transactions_per_temporal_item)[:, None]
        fuzzy_temporal_supports = numerator / denominator
        expected_output = torch.tensor([[0.5, 0.],  # low A, high A
                                        [0.35, 0.05],  # low B, high B
                                        [0.26666665, 0.13333333],  # low C, high C
                                        [0., 1.],  # low D, high D
                                        [0.5, 0.]])  # low E, high E
        assert torch.isclose(fuzzy_temporal_supports, expected_output).all()

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
        denominator = torch.tensor(num_of_possible_transactions_per_temporal_item)[:, None]
        fuzzy_temporal_supports = numerator / denominator
        # start of step 4
        minimum_support = 0.3
        fuzzy_temporal_supports >= minimum_support  # L1 --> low A, low B, high D, and low E
        assert ((fuzzy_temporal_supports >= minimum_support) ==
                torch.tensor([[True, False], [True, False], [False, False], [False, True], [True, False]])).all()

        if torch.count_nonzero(fuzzy_temporal_supports >= minimum_support).item() > 0:  # step 5
            r = 1  # then we continue (with step 6), where r is the number of items in the current itemsets

        L1_row_indices, L1_col_indices = torch.where(fuzzy_temporal_supports >= minimum_support)  # L1 items' indices
        L1_indices = list(zip(L1_row_indices.tolist(), L1_col_indices.tolist()))

        from itertools import combinations

        # step 7: (low A, low B), (low A, high D), (low A, low E), (low B, high D), (low B, low E), (high D, low E)
        C2_indices = list(combinations(L1_indices, r=r + 1))

        # step 8:

        # step 8.1:

        special_idx = granulation.make_dont_care_membership()
        nan_array = np.empty(len(ti_table.terms.keys()))
        nan_array[:] = special_idx
        from copy import deepcopy
        rules = []
        consequences = []
        for candidate in C2_indices:
            antecedents = deepcopy(nan_array)
            for var_idx, term_idx in candidate:
                antecedents[var_idx] = term_idx
            consequences.append(int(0))
            rules.append(Rule(antecedents, consequents=[0]))
        links = make_links_from_antecedents_to_rules(granulation, rules)
        mi = MinimumInference(in_features=len(terms.keys()), out_features=1, consequences=torch.tensor(consequences),
                              links=links)
        antecedents_memberships = granulation(torch.tensor(dataframe[terms.keys()].values).float())
        actual_output = mi.calc_rules_applicability(antecedents_memberships)
        expected_output = torch.tensor([0., 0.5, 0., 0.5, 0.5])
        assert torch.isclose(actual_output[:, 0], expected_output).all()

        # step 8.2:

        scalar_cardinality = actual_output.sum(dim=0)
        expected_scalar_cardinality = torch.tensor([1.5, 1.0, 0.5, 1.25, 0.75, 1.0])
        assert torch.isclose(actual_output.sum(dim=0), expected_scalar_cardinality).all()

        # step 8.3

        # we need to get each temporal item's corresponding starting period
        item_indices_in_each_candidate = [tuple([pair[0] for pair in candidate]) for candidate in C2_indices]
        # (0, 1) means the first and second items in ti_table.terms.keys(), and so on
        assert item_indices_in_each_candidate == [(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4)]

        starting_periods_per_item_in_each_candidate = [[ti_table.starting_periods.values[0, var_idx]
                                                        for var_idx in candidate_indices]
                                                       for candidate_indices in item_indices_in_each_candidate]
        assert starting_periods_per_item_in_each_candidate == [[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]

        # get the maximum starting period within each candidate to calculate fuzzy temporal support
        max_starting_periods = np.array(starting_periods_per_item_in_each_candidate).max(axis=1)

        assert (max_starting_periods == np.array([0, 1, 1, 1, 1, 1])).all()

        num_of_transactions_per_candidate = [ti_table.size_of_transactions_per_time_granule.values[idx:].sum()
                                             for idx in max_starting_periods]
        num_of_transactions_per_candidate = np.array(num_of_transactions_per_candidate)

        assert (num_of_transactions_per_candidate == np.array([5, 2, 2, 2, 2, 2])).all()

        fuzzy_temporal_supports = scalar_cardinality / torch.tensor(num_of_transactions_per_candidate)

        assert torch.isclose(fuzzy_temporal_supports, torch.tensor([0.3, 0.5, 0.25, 0.625, 0.375, 0.5])).all()

        L2_indices = torch.where(fuzzy_temporal_supports >= minimum_support)[0]  # L2 items' indices
