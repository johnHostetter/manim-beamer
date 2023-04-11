import torch
import pygad
import unittest
import numpy as np

from utils.reproducibility import set_rng
from soft.computing.design import expert_design
from soft.fuzzy.sets.continuous import Gaussian
from soft.fuzzy.relation.aggregation import OrderedWeightedAveraging as OWA
from soft.fuzzy.linguistic.summary import Summary, Query, most_quantifier as Q

set_rng(1)
X = torch.rand((100, 2))
antecedents = [Gaussian(4), Gaussian(4)]


def check_initial_population(ga_instance):
    prevent_no_fuzzy_sets(ga_instance)


def prevent_no_fuzzy_sets(ga_instance, offspring_mutation=None):
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
        ga_instance:
        offspring_mutation:

    Returns:

    """
    if offspring_mutation is None:
        population = ga_instance.initial_population  # check that each solution in the population is valid
    else:
        population = offspring_mutation
    indices_that_contain_no_chosen_fuzzy_sets = np.where((population < 0).all(axis=1))[0]
    for row_index_to_change in indices_that_contain_no_chosen_fuzzy_sets:
        col_index_to_change = np.random.choice(population.shape[1])
        valid_gene_choice_indices = np.array(ga_instance.gene_space[col_index_to_change]) >= 0
        valid_gene_choices = np.array(ga_instance.gene_space[col_index_to_change])[valid_gene_choice_indices]
        population[row_index_to_change, col_index_to_change] = np.random.choice(valid_gene_choices)
    if offspring_mutation is None:  # the initial population needs to be updated
        ga_instance.initial_population = ga_instance.population = population


def fitness_function(self, solution, solution_idx):
    global antecedents, X
    candidate = tuple([  # term indices < 0 are reserved for "removed" fuzzy sets
        (variable_index, int(term_index)) for variable_index, term_index in enumerate(solution) if term_index >= 0])
    kb = expert_design(antecedents, rules=[candidate])
    candidate = Summary(kb, quantifier=Q, truth=None)
    query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
    return candidate.degree_of_validity(X, alpha=0.3, query=query).item()


def make_scenario_1():
    terms = [Gaussian(1, centers=[0.8], widths=[0.25]), Gaussian(1, centers=[0.4], widths=[0.25])]
    kb = expert_design(terms, rules=[((0, 0), (1, 0))])  # the 'rule' encodes the linguistic summary
    summary = Summary(kb, Q, None)
    # we want the second attribute to satisfy this
    query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
    X = torch.tensor([[1., 0.5], [0.6, 0.4], [0.1, 0.3], [0.9, 0.7]])
    return X, query, summary


class TestSummary(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_most_quantifier(self):
        """
        Test that the 'most' quantifier fuzzy set behaves as intended.

        Returns:
            None
        """
        assert Q(1.0) == 1.0
        assert Q(0.8) == 1.0
        assert np.isclose(Q(0.7), 0.8)
        assert Q(0.3) == 0.0

    def test_linguistic_quantified_proposition(self):
        elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
        n_inputs = 1
        property_mf = Gaussian(n_inputs, centers=[0.8], widths=[0.4])
        assert property_mf.centers.detach().numpy() == 0.8
        assert property_mf.sigmas.detach().numpy() == 0.4
        mu = property_mf(elements)
        x = mu.sum() / elements.nelement()
        assert torch.isclose(x, torch.tensor(0.7572454810142517))  # compare to ground truth value
        truth_of_proposition = Q(x)
        assert torch.isclose(truth_of_proposition, torch.tensor(0.9145))  # compare to ground truth value

    def test_linguistic_quantified_proposition_with_importance(self):
        elements = torch.tensor([0.7, 0.6, 0.8, 0.9, 0.74, 0.45, 0.64, 0.2])
        n_inputs = 1
        property_mf = Gaussian(n_inputs, centers=[0.8], widths=[0.4])
        importance_mf = Gaussian(n_inputs, centers=[0.6], widths=[0.2])
        assert property_mf.centers.detach().numpy() == 0.8
        assert property_mf.sigmas.detach().numpy() == 0.4
        assert importance_mf.centers.detach().numpy() == 0.6
        assert importance_mf.sigmas.detach().numpy() == 0.2
        property_mu = property_mf(elements)
        importance_mu = importance_mf(elements)
        t_norm_results = property_mu * importance_mu
        assert torch.isclose(t_norm_results.flatten(),
                             torch.tensor([0.7316157, 0.77880085, 0.3678795, 0.09901349, 0.5989963,
                                           0.26497352, 0.8187308, 0.00193045])).all()
        assert torch.isclose(importance_mu.sum(), torch.tensor(4.4135942459106445))
        x = t_norm_results.sum() / importance_mu.sum()
        assert torch.isclose(x, torch.tensor(0.8296958208084106))  # compare to ground truth value
        truth_of_proposition = Q(x)
        assert torch.isclose(truth_of_proposition, torch.tensor(1.0))  # compare to ground truth value

    def test_owa_with_importance(self):
        importance = torch.tensor([0.2, 0.3, 0.1, 0.4])
        assert importance.sum() == 1.0
        in_features = len(importance)
        p = in_features
        x = torch.tensor([0, 0.7, 1.0, 0.2])
        sorted_x = torch.sort(x, descending=True)  # namedtuple with 'values' and 'indices' properties
        assert torch.isclose(sorted_x.values, torch.tensor([1.0000, 0.7000, 0.2000, 0.0000])).all()
        sorted_importance = importance[sorted_x.indices]
        assert torch.isclose(sorted_importance, torch.tensor([0.1000, 0.3000, 0.4000, 0.2000])).all()

        denominator = sorted_importance.sum()
        weights = []
        for j in range(p):
            left_side = Q(sorted_importance[:j + 1].sum() / denominator)
            right_side = Q(sorted_importance[:j].sum() / denominator)
            weights.append((left_side - right_side).item())
        weights = torch.tensor(weights)

        owa = OWA(in_features, weights)
        assert torch.isclose(owa.weights, weights).all()
        assert torch.isclose(owa(x), torch.tensor(0.30))

    def test_summarizer_membership(self):
        """
        The membership of the summarizer should be equal to the minimum membership found across the list of fuzzy sets
        seen in the summarizer argument.

        Returns:
            None
        """
        terms = [Gaussian(1, centers=[0.8], widths=[0.25]), Gaussian(1, centers=[0.4], widths=[0.25])]
        kb = expert_design(terms, rules=[((0, 0), (1, 0))])  # the 'rule' encodes the linguistic summary
        summary = Summary(kb, Q, None)

        x = torch.tensor([[1., 0.5]])
        assert torch.isclose(summary.summarizer_membership(x), torch.tensor(0.5272924900054932))

    def test_summarizer_membership_query(self):
        """
        The membership of the summarizer should be equal to the minimum membership found across the list of fuzzy sets
        seen in the summarizer argument.

        Returns:
            None
        """
        terms = [Gaussian(1, centers=[0.8], widths=[0.25]), Gaussian(1, centers=[0.4], widths=[0.25])]
        kb = expert_design(terms, rules=[((0, 0), (1, 0))])  # the 'rule' encodes the linguistic summary
        summary = Summary(kb, Q, None)

        x = torch.tensor([[1., 0.5]])
        # we want to constrain that the second attribute has to satisfy the following
        query = Query(Gaussian(1, centers=0.3, widths=0.3), 1)
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.5272924900054932))  # it should
        # we want the second attribute to satisfy this
        query = Query(Gaussian(1, centers=0.25, widths=0.3), 1)
        # the given x does not match as well with the (fuzzy) query
        assert torch.isclose(summary.summarizer_membership(x, query), torch.tensor(0.4993517994880676))

    def test_degree_of_truth(self):
        X, query, summary = make_scenario_1()
        assert torch.isclose(summary.degree_of_truth(X, query=query), torch.tensor(0.3612580895423889))

    def test_degree_of_imprecision(self):
        X, query, summary = make_scenario_1()
        assert torch.isclose(summary.degree_of_imprecision(X, alpha=0.3), torch.tensor(1 / 4))

    def test_degree_of_covering(self):
        X, query, summary = make_scenario_1()
        assert torch.isclose(summary.degree_of_covering(X, alpha=0.3, query=query), torch.tensor(2 / 3))

    def test_degree_of_appropriateness(self):
        X, query, summary = make_scenario_1()
        assert torch.isclose(summary.degree_of_appropriateness(X, alpha=0.3, query=query),
                             torch.tensor(0.10416668653488159))

    def test_length(self):
        terms = [Gaussian(1, centers=[0.8], widths=[0.25]), Gaussian(1, centers=[0.4], widths=[0.25])]
        kb = expert_design(terms, rules=[((0, 0), (1, 0))])  # the 'rule' encodes the linguistic summary
        summary = Summary(kb, Q, None)
        assert torch.isclose(summary.length(), torch.tensor(1 / 2))

    def test_degree_of_validity(self):
        X, query, summary = make_scenario_1()
        assert torch.isclose(summary.degree_of_validity(X, alpha=0.3, query=query),
                             torch.tensor(0.3764182925224304))

    def test_prevent_no_fuzzy_sets(self):
        X, query, summary = make_scenario_1()
        gene_space = [list(range(-1, max_terms + 1)) for max_terms in summary.kb.intra_dimensions()]
        assert gene_space == [[-1, 0, 1], [-1, 0, 1]]
        ga_instance = pygad.GA(num_generations=10,
                               num_parents_mating=2,
                               fitness_func=fitness_function,
                               sol_per_pop=10,
                               num_genes=summary.kb.variable_dimensions(),
                               mutation_num_genes=1,
                               gene_space=gene_space,
                               on_start=check_initial_population,
                               on_mutation=prevent_no_fuzzy_sets)
        # the bottom row is an invalid combination (i.e., all negatives)
        expected_population = np.array([[1., -1.],
                                        [0., 0.],
                                        [-1., 0.],
                                        [1., -1.],
                                        [0., 1.],
                                        [0., 1.],
                                        [1., 0.],
                                        [1., -1.],
                                        [-1., 1.],
                                        [-1., -1.]])
        assert (ga_instance.population == expected_population).all()
        assert (ga_instance.initial_population == expected_population).all()

        prevent_no_fuzzy_sets(ga_instance)

        # the bottom row has been corrected
        expected_population = np.array([[1., -1.],
                                        [0., 0.],
                                        [-1., 0.],
                                        [1., -1.],
                                        [0., 1.],
                                        [0., 1.],
                                        [1., 0.],
                                        [1., -1.],
                                        [-1., 1.],
                                        [1., -1.]])
        assert (ga_instance.population == expected_population).all()
        assert (ga_instance.initial_population == expected_population).all()

    def test_genetic_algorithm_summary_search(self):
        X, query, summary = make_scenario_1()
        gene_space = [list(range(-1, max_terms + 1)) for max_terms in summary.kb.intra_dimensions()]
        assert gene_space == [[-1, 0, 1], [-1, 0, 1]]
        ga_instance = pygad.GA(num_generations=10,
                               num_parents_mating=2,
                               fitness_func=fitness_function,
                               sol_per_pop=10,
                               num_genes=summary.kb.variable_dimensions(),
                               mutation_num_genes=1,
                               gene_space=gene_space,
                               on_start=check_initial_population,
                               on_mutation=prevent_no_fuzzy_sets)

        ga_instance.run()
        print('Initial population:')
        print(ga_instance.initial_population)
        print('Population after {} generations:'.format(ga_instance.num_generations))
        print(ga_instance.population)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print('Parameters of the best solution : {solution}'.format(solution=solution))
        print('Fitness value of the best solution = {solution_fitness}'.format(solution_fitness=solution_fitness))
        print('Index of the best solution : {solution_idx}'.format(solution_idx=solution_idx))
