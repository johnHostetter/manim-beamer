"""
Test the linguistic summary code implementation.
"""
import unittest
from typing import Tuple, List

import torch
import pygad
import numpy as np

from YACS.yacs import Config
from soft.utilities.reproducibility import set_rng, load_configuration
from soft.computing.design import expert_design
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.logic.rules.creation import Rule
from soft.fuzzy.relation.tnorm import AlgebraicProduct
from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA
from soft.fuzzy.linguistic.summary import Summary, Query, most_quantifier as Q


def check_initial_population(ga_instance: pygad.GA) -> None:
    """
    Wrapper method to modify the initial population, so it does not violate constraints.

    Args:
        ga_instance: The genetic algorithm instance.

    Returns:
        None
    """
    prevent_no_fuzzy_sets(ga_instance)


def prevent_no_fuzzy_sets(
    ga_instance: pygad.GA, offspring_mutation: np.ndarray = None
) -> None:
    """
    This function checks that the population does not contain
    an invalid selection of gene values. Specifically, a row,
    or candidate, cannot contain all negative values. This is
    because negative values have a special meaning in this code;
    the presence of a negative value means that the attribute/variable
    should be disregarded (i.e., feature selection).

    Hence, a row of all negative values essentially amounts to
    no features (i.e., fuzzy sets) selected, which is not allowed.

    Args:
        ga_instance: The genetic algorithm instance.
        offspring_mutation: The offspring from the mutation operation.

    Returns:

    """
    if offspring_mutation is None:
        # check that each solution in the population is valid
        population = ga_instance.initial_population
    else:
        population = offspring_mutation
    indices_that_contain_no_chosen_fuzzy_sets = np.where((population < 0).all(axis=1))[
        0
    ]
    for row_index_to_change in indices_that_contain_no_chosen_fuzzy_sets:
        col_index_to_change = np.random.choice(population.shape[1])
        valid_gene_choice_indices = (
            np.array(ga_instance.gene_space[col_index_to_change]) >= 0
        )
        valid_gene_choices = np.array(ga_instance.gene_space[col_index_to_change])[
            valid_gene_choice_indices
        ]
        population[row_index_to_change, col_index_to_change] = np.random.choice(
            valid_gene_choices
        )
    if offspring_mutation is None:  # the initial population needs to be updated
        ga_instance.initial_population = ga_instance.population = population


def fitness_function_factory(
    input_data: torch.Tensor, antecedents: List[Gaussian], config: Config
) -> callable:
    """
    Factory design pattern to initialize the fitness_function's environment with the necessary
    variables, as the fitness_function expects a certain signature that cannot be modified.

    Args:
        input_data: The input data to be used for the fitness function.
        antecedents: The antecedent fuzzy sets.
        config: The configuration object.

    Returns:
        fitness_function
    """

    def fitness_function(
        ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int
    ) -> float:
        """
        The fitness function for the genetic algorithm search.

        Args:
            self: The genetic algorithm instance.
            solution: The solution to evaluate.
            solution_idx: The index of the solution.

        Returns:
            The fitness of the solution.
        """
        print(f"{ga_instance}: {solution_idx}")
        candidate = (  # term indices < 0 are reserved for "removed" fuzzy sets
            (variable_index, int(term_index))
            for variable_index, term_index in enumerate(solution)
            if term_index >= 0
        )
        rule = Rule(
            premise=candidate, consequence=frozenset(), implication=AlgebraicProduct
        )
        knowledge_base = expert_design(
            antecedents, consequents=[], rules=[rule], config=config
        )
        candidate = Summary(knowledge_base, quantifier=Q, truth=None)
        query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
        return candidate.degree_of_validity(input_data, alpha=0.3, query=query).item()

    return fitness_function


def scenario_1() -> Tuple[torch.Tensor, Query, Summary]:
    """
    Create a simple test scenario that has only a few terms and a couple data observations.

    Returns:
        The input data, the query, and the summary.
    """
    configuration = load_configuration()
    terms = [
        Gaussian(1, centers=[0.8], widths=[0.25]),
        Gaussian(1, centers=[0.4], widths=[0.25]),
    ]
    rule = Rule(
        premise=frozenset({(0, 0), (1, 0)}),
        consequence=frozenset(),
        implication=AlgebraicProduct,
    )
    knowledge_base = expert_design(  # the 'rule' encodes the linguistic summary
        terms, consequents=[], rules=[rule], config=configuration
    )
    summary = Summary(knowledge_base, Q, None)
    # we want the second attribute to satisfy this
    query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
    input_data = torch.tensor([[1.0, 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
    return input_data, query, summary


def scenario_2() -> Tuple[torch.Tensor, Query, Summary, List[Gaussian]]:
    """
    Create a larger but simpler test scenario that has only a some terms and more data observations.

    Returns:
        The input data, the query, the summary, and the list of antecedent fuzzy sets.
    """
    configuration = load_configuration()
    terms = [
        Gaussian(1, centers=[0.8], widths=[0.25]),
        Gaussian(1, centers=[0.4], widths=[0.25]),
    ]  # terms for the linguistic summary
    rule = Rule(
        premise=frozenset({(0, 0), (1, 0)}),
        consequence=frozenset(),
        implication=AlgebraicProduct,
    )
    knowledge_base = expert_design(  # the 'rule' encodes the linguistic summary
        terms, consequents=[], rules=[rule], config=configuration
    )
    summary = Summary(knowledge_base, Q, None)
    # we want the second attribute to satisfy this
    query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
    input_data = torch.rand((100, 2))
    antecedents = [Gaussian(4), Gaussian(4)]
    return input_data, query, summary, antecedents


class TestSummary(unittest.TestCase):
    """
    Test the Summary class, which implements the linguistic summarization of data approach.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = load_configuration()

    def test_most_quantifier(self) -> None:
        """
        Test that the 'most' quantifier fuzzy set behaves as intended.

        Returns:
            None
        """
        assert Q(1.0) == 1.0
        assert Q(0.8) == 1.0
        assert np.isclose(Q(0.7), 0.8)
        assert Q(0.3) == 0.0

    def test_linguistic_quantified_proposition(self) -> None:
        """
        Test that a linguistically quantified proposition (e.g., 'most') is correctly calculated.

        Returns:
            None
        """
        elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
        n_inputs = 1
        property_mf = Gaussian(n_inputs, centers=[0.8], widths=[0.4])
        assert property_mf.centers.detach().numpy() == 0.8
        assert property_mf.sigmas.detach().numpy() == 0.4
        elements_membership = property_mf(elements)
        element = elements_membership.sum() / elements.nelement()
        # compare to ground truth value
        assert torch.isclose(element, torch.tensor(0.7572454810142517))
        truth_of_proposition = Q(element)
        assert torch.isclose(
            truth_of_proposition, torch.tensor(0.9145)
        )  # compare to ground truth value

    def test_linguistic_quantified_proposition_with_importance(self) -> None:
        """
        Test that a linguistically quantified proposition (e.g., 'most') is correctly calculated
        when there is an additional factor of 'importance' that weighs the calculation.

        Returns:
            None
        """
        elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
        property_mf = Gaussian(in_features=1, centers=[0.8], widths=[0.4])
        importance_mf = Gaussian(in_features=1, centers=[0.6], widths=[0.2])
        assert property_mf.centers.detach().numpy() == 0.8
        assert property_mf.sigmas.detach().numpy() == 0.4
        assert importance_mf.centers.detach().numpy() == 0.6
        assert importance_mf.sigmas.detach().numpy() == 0.2
        property_mu = property_mf(elements)
        importance_mu = importance_mf(elements)
        t_norm_results = property_mu * importance_mu
        assert torch.isclose(
            t_norm_results.flatten(),
            torch.tensor(
                [
                    0.7316157,
                    0.77880085,
                    0.3678795,
                    0.09901349,
                    0.5989963,
                    0.26497352,
                    0.8187308,
                    0.00193045,
                ]
            ),
        ).all()
        assert torch.isclose(importance_mu.sum(), torch.tensor(4.4135942459106445))
        element = t_norm_results.sum() / importance_mu.sum()
        # compare to ground truth value
        assert torch.isclose(element, torch.tensor(0.8296958208084106))
        truth_of_proposition = Q(element)
        assert torch.isclose(
            truth_of_proposition, torch.tensor(1.0)
        )  # compare to ground truth value

    def test_owa_with_importance(self) -> None:
        """
        Test that the ordered weighted averaging for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        importance = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert importance.sum() == 1.0
        in_features = len(importance)
        element = torch.tensor([0, 0.7, 1.0, 0.2])
        sorted_x = torch.sort(
            element, descending=True
        )  # namedtuple with 'values' and 'indices' properties
        assert torch.isclose(
            sorted_x.values, torch.tensor([1.0000, 0.7000, 0.2000, 0.0000])
        ).all()
        sorted_importance = importance[sorted_x.indices]
        assert torch.isclose(
            sorted_importance, torch.tensor([0.1000, 0.3000, 0.4000, 0.2000])
        ).all()

        denominator = sorted_importance.sum()
        weights = []
        for j in range(in_features):
            left_side = Q(sorted_importance[: j + 1].sum() / denominator)
            right_side = Q(sorted_importance[:j].sum() / denominator)
            weights.append((left_side - right_side).item())
        weights = torch.tensor(weights)

        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert torch.isclose(owa(element), torch.tensor(0.30))

    def test_summarizer_membership(self) -> None:
        """
        The membership of the summarizer should be equal to the minimum membership found
        across the list of fuzzy sets seen in the summarizer argument.

        Returns:
            None
        """
        terms = [
            Gaussian(1, centers=[0.8], widths=[0.25]),
            Gaussian(1, centers=[0.4], widths=[0.25]),
        ]
        rule = Rule(
            premise=frozenset({(0, 0), (1, 0)}),
            consequence=frozenset(),
            implication=AlgebraicProduct,
        )
        knowledge_base = expert_design(  # the 'rule' encodes the linguistic summary
            terms, consequents=[], rules=[rule], config=self.config
        )
        summary = Summary(knowledge_base, Q, None)

        element = torch.tensor([[1.0, 0.5]])
        assert torch.isclose(
            summary.summarizer_membership(element), torch.tensor(0.5272924900054932)
        )

    def test_summarizer_membership_query(self) -> None:
        """
        The membership of the summarizer should be equal to the minimum membership found
        across the list of fuzzy set seen in the summarizer argument.

        Returns:
            None
        """
        terms = [
            Gaussian(1, centers=[0.8], widths=[0.25]),
            Gaussian(1, centers=[0.4], widths=[0.25]),
        ]
        rule = Rule(
            premise=frozenset({(0, 0), (1, 0)}),
            consequence=frozenset(),
            implication=AlgebraicProduct,
        )
        knowledge_base = expert_design(  # the 'rule' encodes the linguistic summary
            terms, consequents=[], rules=[rule], config=self.config
        )
        summary = Summary(knowledge_base, Q, None)

        element = torch.tensor([[1.0, 0.5]])
        # we want to constrain that the second attribute has to satisfy the following
        query = Query(Gaussian(1, centers=0.3, widths=0.3), 1)
        assert torch.isclose(
            summary.summarizer_membership(element, query),
            torch.tensor(0.5272924900054932),
        )  # it should
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
        # the given element does not match as well with the (fuzzy) query
        assert torch.isclose(
            summary.summarizer_membership(element, query),
            torch.tensor(0.4993517994880676),
        )

    def test_degree_of_truth(self) -> None:
        """
        Test that the degree of truth for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        input_data, query, summary = scenario_1()
        assert torch.isclose(
            summary.degree_of_truth(input_data, query=query),
            torch.tensor(0.3612580895423889),
        )

    def test_degree_of_imprecision(self) -> None:
        """
        Test that the degree of imprecision for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        input_data, _, summary = scenario_1()
        assert torch.isclose(
            summary.degree_of_imprecision(input_data, alpha=0.3), torch.tensor(1 / 4)
        )

    def test_degree_of_covering(self) -> None:
        """
        Test that the degree of covering for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        input_data, query, summary = scenario_1()
        assert torch.isclose(
            summary.degree_of_covering(input_data, alpha=0.3, query=query),
            torch.tensor(2 / 3),
        )

    def test_degree_of_appropriateness(self) -> None:
        """
        Test that the degree of appropriateness for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        input_data, query, summary = scenario_1()
        assert torch.isclose(
            summary.degree_of_appropriateness(input_data, alpha=0.3, query=query),
            torch.tensor(0.10416668653488159),
        )

    def test_length(self) -> None:
        """
        Test that the length of a linguistic summary is correctly calculated.

        Returns:
            None
        """
        terms = [
            Gaussian(1, centers=[0.8], widths=[0.25]),
            Gaussian(1, centers=[0.4], widths=[0.25]),
        ]
        # the 'rule' encodes the linguistic summary
        rule = Rule(
            premise=frozenset({(0, 0), (1, 0)}),
            consequence=frozenset(),
            implication=AlgebraicProduct,
        )
        knowledge_base = expert_design(
            terms, consequents=[], rules=[rule], config=self.config
        )
        summary = Summary(knowledge_base, Q, None)
        assert torch.isclose(summary.length(), torch.tensor(1 / 2))

    def test_degree_of_validity(self) -> None:
        """
        Test that the degree of validity for a linguistic summary is correctly calculated.

        Returns:
            None
        """
        input_data, query, summary = scenario_1()
        assert torch.isclose(
            summary.degree_of_validity(input_data, alpha=0.3, query=query),
            torch.tensor(0.3764182925224304),
        )

    def test_prevent_no_fuzzy_sets(self) -> None:
        """
        Test that no candidate in the genetic algorithm's population has selected no fuzzy sets.

        Returns:
            None
        """
        set_rng(0)
        dataset, _, summary, linguistic_terms = scenario_2()
        gene_space = [
            list(range(-1, max_terms + 1))
            for max_terms in summary.knowledge_base.intra_dimensions(is_input=True)
        ]
        assert gene_space == [[-1, 0, 1], [-1, 0, 1]]
        ga_instance = pygad.GA(
            num_generations=10,
            num_parents_mating=2,
            fitness_func=fitness_function_factory(
                input_data=dataset, antecedents=linguistic_terms, config=self.config
            ),
            sol_per_pop=10,
            num_genes=summary.knowledge_base.variable_dimensions(is_input=True),
            mutation_num_genes=1,
            gene_space=gene_space,
            on_start=check_initial_population,
            on_mutation=prevent_no_fuzzy_sets,
        )
        # the bottom row is an invalid combination (i.e., all negatives)
        ga_instance.population = ga_instance.initial_population = np.array(
            [
                [1.0, -1.0],
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, -1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
            ]
        )
        # assert (ga_instance.population == expected_population).all()
        # assert (ga_instance.initial_population == expected_population).all()

        prevent_no_fuzzy_sets(ga_instance)

        # the bottom row has been corrected
        expected_population = np.array(
            [
                [1.0, -1.0],
                [0.0, 0.0],
                [-1.0, 0.0],
                [1.0, -1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
            ]
        )
        assert (ga_instance.population == expected_population).all()
        assert (ga_instance.initial_population == expected_population).all()

    def test_genetic_algorithm_summary_search(self) -> None:
        """
        Test the genetic algorithm search for a linguistic summary.

        Returns:
            None
        """
        set_rng(0)
        dataset, _, summary, linguistic_terms = scenario_2()
        gene_space = [
            list(range(-1, max_terms + 1))
            for max_terms in summary.knowledge_base.intra_dimensions(is_input=True)
        ]
        assert gene_space == [[-1, 0, 1], [-1, 0, 1]]
        ga_instance = pygad.GA(
            num_generations=10,
            num_parents_mating=2,
            fitness_func=fitness_function_factory(
                input_data=dataset, antecedents=linguistic_terms, config=self.config
            ),
            sol_per_pop=10,
            num_genes=summary.knowledge_base.variable_dimensions(is_input=True),
            mutation_num_genes=1,
            gene_space=gene_space,
            on_start=check_initial_population,
            on_mutation=prevent_no_fuzzy_sets,
        )

        ga_instance.run()
        print("Initial population:")
        print(ga_instance.initial_population)
        print(f"Population after {ga_instance.num_generations} generations:")
        print(ga_instance.population)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print(f"Parameters of the best solution : {solution}".format(solution=solution))
        print(f"Fitness value of the best solution = {solution_fitness}")
        print(f"Index of the best solution : {solution_idx}")
