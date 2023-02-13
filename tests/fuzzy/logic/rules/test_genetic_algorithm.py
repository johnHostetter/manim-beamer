import pygad
import torch
import unittest

from soft.computing.blueprints import clip_only
from soft.fuzzy.logic.control.tsk import ZeroOrderTSK
from soft.fuzzy.logic.control.evolutionary import fitness_function_factory, make_kb_from_ga_solution


class TestGeneticAlgorithmRuleSearch(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = torch.rand(10, 5)
        self.output_data = torch.rand(10)

    def test_initial_population(self):
        so = clip_only(self.input_data, self.output_data)
        variables = so.start()

        gene_space = [list(range(0, variable.centers.shape[0])) for variable in variables]

        ga_instance = pygad.GA(num_generations=10,
                               num_parents_mating=2,
                               fitness_func=fitness_function_factory(variables, self.input_data, self.output_data),
                               sol_per_pop=10,
                               num_genes=len(variables),
                               mutation_num_genes=1,
                               gene_space=gene_space)

        ga_instance.run()
        print('Initial population:')
        print(ga_instance.initial_population)
        print('Population after {} generations:'.format(ga_instance.num_generations))
        print(ga_instance.population)
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print('Parameters of the best solution : {solution}'.format(solution=solution))
        print('Fitness value of the best solution = {solution_fitness}'.format(solution_fitness=solution_fitness))
        print('Index of the best solution : {solution_idx}'.format(solution_idx=solution_idx))

        kb = make_kb_from_ga_solution(variables, solution)
        tsk = ZeroOrderTSK(out_features=1, knowledge_base=kb, learning_rate=1e-3, input_trainable=True)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(tsk(self.input_data), self.output_data).item()
        print(loss)
