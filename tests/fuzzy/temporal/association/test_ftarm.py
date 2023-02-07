import torch
import unittest
import numpy as np
import pandas as pd

from utils.reproducibility import set_rng
from soft.computing.design import expert_design
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.graph.organizing import stack_granules
from examples.fuzzy.temporal.association.ftarm.demo_ftarm import make_example
from soft.fuzzy.temporal.association.ftarm import make_candidates_inference_engine, TemporalInformationTable as TI, \
    FuzzyTemporalAssocationRuleMining as FTARM, AssociationRule

set_rng(5)


def big_data_example(seed):
    set_rng(seed)
    dataframe = pd.DataFrame(np.random.rand(4000, 4))
    dataframe['date'] = 0
    variables = {
        0: Gaussian(in_features=4),
        1: Gaussian(in_features=4),
        2: Gaussian(in_features=4),
        3: Gaussian(in_features=4),
    }

    kb = expert_design(variables.values())
    return dataframe, kb


class TestFTARM(unittest.TestCase):
    def test_fuzzy_representation(self):
        dataframe, kb = make_example()
        input_granulation = kb.graph.vs.find(source_eq=stack_granules)['name']
        cols = sorted(set(dataframe.columns) - {'date'})
        mus = input_granulation(torch.tensor(dataframe[cols].values).float())
        expected_membership = torch.tensor([[1.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                            [0.0, 0.0, 0.0, 0.0, 2 / 3, 1 / 3, 0.0, 0.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                            [0.5, 0.0, 0.75, 0.25, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
        assert (torch.isclose(mus.reshape(5, 10), expected_membership)).all()

    def test_temporal_information_table(self):
        dataframe, kb = make_example()
        attribute_names = [col for col in dataframe.columns if col != 'date']
        ti_table = TI(dataframe, variables=attribute_names)
        # temporal item A occurs in the first transaction, B occurs in the second, C occurs in the first, and so on
        assert (ti_table.first_transaction_indices == np.array([0, 1, 0, 3, 4])).all()
        # temporal items D and E come in the second time period, all others occur in the first time period
        assert (ti_table.starting_periods.values == np.array([[0, 0, 0, 1, 1]])).all()

        # now checking that FTARM creates the same TI Table as above
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)
        # temporal item A occurs in the first transaction, B occurs in the second, C occurs in the first, and so on
        assert (ftarm.ti_table.first_transaction_indices == np.array([0, 1, 0, 3, 4])).all()
        # temporal items D and E come in the second time period, all others occur in the first time period
        assert (ftarm.ti_table.starting_periods.values == np.array([[0, 0, 0, 1, 1]])).all()

    def test_step_2(self):
        dataframe, kb = make_example()
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)
        actual_scalar_cardinality = ftarm.scalar_cardinality()
        expected_scalar_cardinality = torch.tensor([2.5, 0.0, 1.75, 0.25, 4 / 3, 2 / 3, 0.0, 2.0, 1.0, 0.0])
        assert torch.isclose(actual_scalar_cardinality.flatten(), expected_scalar_cardinality).all()

    def test_step_3(self):
        dataframe, kb = make_example()
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)
        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports()
        expected_output = torch.tensor([[0.5, 0.],  # low A, high A
                                        [0.35, 0.05],  # low B, high B
                                        [0.26666665, 0.13333333],  # low C, high C
                                        [0., 1.],  # low D, high D
                                        [0.5, 0.]])  # low E, high E
        assert torch.isclose(actual_fuzzy_temporal_supports, expected_output).all()

    def test_step_4(self):
        dataframe, kb = make_example()
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)
        # start of step 4
        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports()
        actual_fuzzy_temporal_supports >= ftarm.minimum_support  # L1 --> low A, low B, high D, and low E
        assert ((actual_fuzzy_temporal_supports >= ftarm.minimum_support) ==
                torch.tensor([[True, False], [True, False], [False, False], [False, True], [True, False]])).all()

        C2_indices = ftarm.make_candidates()

        # step 7: (low A, low B), (low A, high D), (low A, low E), (low B, high D), (low B, low E), (high D, low E)
        assert C2_indices == [
            ((0, 0), (1, 0)), ((0, 0), (3, 1)), ((0, 0), (4, 0)),
            ((1, 0), (3, 1)), ((1, 0), (4, 0)), ((3, 1), (4, 0))
        ]

    def test_candidate_fuzzy_representation_functions(self):
        dataframe, kb = make_example()
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()

        # step 8.1:

        mi = make_candidates_inference_engine(ftarm, candidates=C2_indices)

        expected_links = torch.tensor(
            [[[1., 1., 1., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.]],
             [[1., 0., 0., 1., 1., 0.],
              [0., 0., 0., 0., 0., 0.]],
             [[0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 0., 0.]],
             [[0., 0., 0., 0., 0., 0.],
              [0., 1., 0., 1., 0., 1.]],
             [[0., 0., 1., 0., 1., 1.],
              [0., 0., 0., 0., 0., 0.]]]
        )

        assert (mi.links_between_antecedents_and_rules == expected_links).all()

        expected_offset = torch.tensor(
            [[0., 0., 0., 1., 1., 1.],
             [0., 1., 1., 0., 0., 1.],
             [1., 1., 1., 1., 1., 1.],
             [1., 0., 1., 0., 1., 0.],
             [1., 1., 0., 1., 0., 0.]]
        )

        assert (mi.offset == expected_offset).all()

        actual_antecedents_memberships = ftarm.granulation(
            torch.tensor(dataframe[ftarm.variables].values).float())

        expected_memberships = torch.tensor([[[1.0000, 0.0000],
                                              [0.0000, 0.0000],
                                              [2 / 3, 1 / 3],
                                              [0.0000, 0.0000],
                                              [0.0000, 0.0000]],
                                             [[0.5000, 0.0000],
                                              [0.5000, 0.0000],
                                              [0.0000, 0.0000],
                                              [0.0000, 0.0000],
                                              [0.0000, 0.0000]],
                                             [[0.0000, 0.0000],
                                              [0.0000, 0.0000],
                                              [2 / 3, 1 / 3],
                                              [0.0000, 0.0000],
                                              [0.0000, 0.0000]],
                                             [[0.5000, 0.0000],
                                              [0.5000, 0.0000],
                                              [0.0000, 0.0000],
                                              [0.0000, 1.0000],
                                              [0.0000, 0.0000]],
                                             [[0.5000, 0.0000],
                                              [0.7500, 0.2500],
                                              [0.0000, 0.0000],
                                              [0.0000, 1.0000],
                                              [1.0000, 0.0000]]])

        assert torch.isclose(actual_antecedents_memberships, expected_memberships, equal_nan=True).all()

        actual_intermediate_output = mi.calc_intermediate_output(actual_antecedents_memberships)
        expected_intermediate_output = torch.tensor([[[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000],
                                                      [1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000]],

                                                     [[0.5000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000],
                                                      [0.5000, 1.0000, 1.0000, 0.5000, 0.5000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000],
                                                      [1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000]],

                                                     [[0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 1.0000],
                                                      [0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000],
                                                      [1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000]],

                                                     [[0.5000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000],
                                                      [0.5000, 1.0000, 1.0000, 0.5000, 0.5000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000]],

                                                     [[0.5000, 0.5000, 0.5000, 1.0000, 1.0000, 1.0000],
                                                      [0.7500, 1.0000, 1.0000, 0.7500, 0.7500, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
                                                      [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]]])

        assert torch.isclose(actual_intermediate_output, expected_intermediate_output).all()

        actual_rules_applicability = mi.calc_rules_applicability(actual_antecedents_memberships)
        expected_output = torch.tensor([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.50, 0.50, 0.00, 0.50, 0.00, 0.00],
                                        [0.50, 0.50, 0.50, 0.75, 0.75, 1.00]])
        assert torch.isclose(actual_rules_applicability, expected_output).all()

    def test_candidate_fuzzy_representation_ftarm(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()

        # step 8.1:

        # now checking that FTARM calculates the same as above
        expected_output = torch.tensor([[0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.50, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                                        [0.50, 0.50, 0.00, 0.50, 0.00, 0.00],
                                        [0.50, 0.50, 0.50, 0.75, 0.75, 1.00]])

        assert torch.isclose(ftarm.fuzzy_representation(C2_indices), expected_output).all()

    def test_candidate_scalar_cardinality(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()

        # step 8.2:

        actual_scalar_cardinality = ftarm.scalar_cardinality(C2_indices)
        expected_scalar_cardinality = torch.tensor([1.5, 1.0, 0.5, 1.25, 0.75, 1.0])
        assert torch.isclose(actual_scalar_cardinality, expected_scalar_cardinality).all()

    def test_candidate_fuzzy_temporal_supports(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()
        # the following candidate order is assumed for the following assertions
        assert C2_indices == [
            ((0, 0), (1, 0)), ((0, 0), (3, 1)),
            ((0, 0), (4, 0)), ((1, 0), (3, 1)),
            ((1, 0), (4, 0)), ((3, 1), (4, 0))
        ]

        # step 8.3

        # we need to get each temporal item's corresponding starting period
        item_indices_in_each_candidate = [tuple([pair[0] for pair in candidate]) for candidate in C2_indices]
        # (0, 1) means the first and second items in ti_table.terms.keys(), and so on
        assert item_indices_in_each_candidate == [(0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4)]

        starting_periods_per_item_in_each_candidate = [[ftarm.ti_table.starting_periods.values[0, var_idx]
                                                        for var_idx in candidate_indices]
                                                       for candidate_indices in item_indices_in_each_candidate]
        assert starting_periods_per_item_in_each_candidate == [[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [1, 1]]

        # get the maximum starting period within each candidate to calculate fuzzy temporal support
        max_starting_periods = np.array(starting_periods_per_item_in_each_candidate).max(axis=1)

        assert (max_starting_periods == np.array([0, 1, 1, 1, 1, 1])).all()

        num_of_transactions_per_candidate = [ftarm.ti_table.size_of_transactions_per_time_granule.values[idx:].sum()
                                             for idx in max_starting_periods]
        num_of_transactions_per_candidate = np.array(num_of_transactions_per_candidate)

        assert (num_of_transactions_per_candidate == np.array([5, 2, 2, 2, 2, 2])).all()

        fuzzy_temporal_supports = ftarm.scalar_cardinality(C2_indices) / torch.tensor(num_of_transactions_per_candidate)
        expected_fuzzy_temporal_supports = torch.tensor([0.3, 0.5, 0.25, 0.625, 0.375, 0.5])

        assert torch.isclose(fuzzy_temporal_supports, expected_fuzzy_temporal_supports).all()

        # now checking that FTARM calculates the same as above

        assert torch.isclose(ftarm.fuzzy_temporal_supports(C2_indices), expected_fuzzy_temporal_supports).all()

        L2_indices = torch.where(fuzzy_temporal_supports >= ftarm.minimum_support)[0]  # L2 items' indices

        assert (L2_indices == torch.tensor([0, 1, 3, 4, 5])).all()

    def test_make_candidate_3_itemsets(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()
        C3_indices = ftarm.make_candidates(C2_indices)
        expected_candidate_indices = [{(1, 0), (3, 1), (0, 0)}, {(1, 0), (4, 0), (3, 1)}]
        assert C3_indices == expected_candidate_indices

        actual_fuzzy_temporal_supports = ftarm.fuzzy_temporal_supports(C3_indices)
        expected_fuzzy_temporal_supports = torch.tensor([0.5000, 0.3750])
        assert torch.isclose(actual_fuzzy_temporal_supports, expected_fuzzy_temporal_supports).all()

    def test_make_candidate_4_itemsets(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)

        C2_indices = ftarm.make_candidates()
        C3_indices = ftarm.make_candidates(C2_indices)
        C4_indices = ftarm.make_candidates(C3_indices)
        expected_candidate_indices = None
        assert C4_indices == expected_candidate_indices

    def test_execute(self):
        dataframe, linguistic_variables = make_example()
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 0), (1, 0)), ((0, 0), (3, 1)), ((0, 0), (4, 0)),
            ((1, 0), (3, 1)), ((1, 0), (4, 0)), ((3, 1), (4, 0))
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [{(1, 0), (3, 1), (0, 0)}, {(1, 0), (4, 0), (3, 1)}]

    def test_find_association_rules(self):
        dataframe, kb = make_example()
        ftarm = FTARM(dataframe, kb, config={}, minimum_support=0.3, minimum_confidence=0.8)
        _ = ftarm.find_candidates()  # required to build the lattice structure which is later used to find rules
        actual_rules = ftarm.find_association_rules()

        expected_rules = {
            AssociationRule(antecedents=frozenset({(1, 0), (4, 0)}), consequents=frozenset({(3, 1)}), confidence=1.0),
            AssociationRule(antecedents=frozenset({(1, 0)}), consequents=frozenset({(3, 1)}), confidence=1.0),
            AssociationRule(antecedents=frozenset({(4, 0)}), consequents=frozenset({(3, 1)}), confidence=1.0),
            AssociationRule(antecedents=frozenset({(1, 0), (0, 0)}), consequents=frozenset({(3, 1)}), confidence=1.0),
            AssociationRule(antecedents=frozenset({(0, 0)}), consequents=frozenset({(3, 1)}), confidence=1.0),
            AssociationRule(antecedents=frozenset({(3, 1), (0, 0)}), consequents=frozenset({(1, 0)}), confidence=1.0)
        }

        for actual_rule, expected_rule in zip(actual_rules, expected_rules):
            assert actual_rule.antecedents == expected_rule.antecedents
            assert actual_rule.consequents == expected_rule.consequents
            assert actual_rule.confidence == expected_rule.confidence

        # fig, axs = plt.subplots()
        # axs = axs
        #
        # import igraph as ig
        #
        # g = kb.graph.subgraph(kb.graph.vs.select(frequent_eq=True))
        # ig.plot(
        #     # ig.VertexCover(kb.graph, kb.graph.vs.select(frequent_eq=True)),
        #     g,
        #     mark_groups=True, palette=ig.RainbowPalette(),
        #     layout='circle', edge_width=0.5, target=axs, opacity=0.7,
        #     vertex_size=(np.array(kb.graph.authority_score()) / 3) + 0.1,
        #     edge_size=kb.graph.es['weight']
        # )
        # plt.axis('off')
        # plt.show()

    def test_execute_big_data_0(self):
        """
        This unit test was introduced to identify that frequent itemsets were not being created that contained more
        than one linguistic variable assignment. Also, some operations originally would result in an empty list
        being returned that the above unit tests do not capture.
        Returns:
            None
        """
        dataframe, linguistic_variables = big_data_example(seed=0)
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 3), (1, 0)), ((0, 3), (1, 1)), ((0, 3), (1, 3)), ((0, 3), (3, 2)),
            ((1, 0), (3, 2)), ((1, 1), (3, 2)), ((1, 3), (3, 2))
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [{(1, 1), (0, 3), (3, 2)}, {(1, 0), (3, 2), (0, 3)}, {(3, 2), (0, 3), (1, 3)}]

    def test_execute_big_data_5(self):
        """
        This unit test was introduced to identify that frequent itemsets were not being created that contained more
        than one linguistic variable assignment. Also, some operations originally would result in an empty list
        being returned that the above unit tests do not capture.
        Returns:
            None
        """
        dataframe, linguistic_variables = big_data_example(seed=5)
        ftarm = FTARM(dataframe, linguistic_variables, config={}, minimum_support=0.3, minimum_confidence=0.8)
        candidates_family = ftarm.find_candidates()
        # the first item in the family should match the expected 2-itemsets
        assert candidates_family[0] == [
            ((0, 1), (2, 2)), ((0, 1), (2, 3)), ((0, 1), (3, 0)), ((0, 3), (2, 2)),
            ((0, 3), (2, 3)), ((0, 3), (3, 0)), ((2, 2), (3, 0)), ((2, 3), (3, 0))
        ]
        # the second item in the family should match the expected 3-itemsets
        assert candidates_family[1] == [
            {(0, 1), (2, 3), (3, 0)}, {(2, 3), (0, 3), (3, 0)},
            {(0, 3), (3, 0), (2, 2)}, {(0, 1), (3, 0), (2, 2)}
        ]
