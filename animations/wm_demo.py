from typing import Set

import torch

from manim import *
from colors import *
from soft.datasets import SupervisedDataset
from soft.fuzzy.logic.rules import LinguisticVariables, Rule
from soft.utilities.reproducibility import set_rng, load_configuration
from soft.fuzzy.sets.continuous.impl import Gaussian
from soft.fuzzy.unsupervised.cluster.online.ecm import (
    apply_evolving_clustering_method as ECM, LabeledClusters
)
from soft.fuzzy.logic.rules.creation.wang_mendel import wang_mendel_method
from soft.fuzzy.unsupervised.granulation.online.clip import (
    apply_categorical_learning_induced_partitioning as CLIP
)
from animations.common import get_data_and_env, display_cart_pole

set_rng(0)


class WMDemo(Scene):
    def __init__(self, **kwargs):
        super().__init__()
        background = ImageMobject('background.png').scale(2).set_color('#FFFFFF')
        self.add(background)

    def make_fuzzy_set(self, ax, center, width, label=None):
        gaussian_graph = ax.plot(
            lambda x: Gaussian(centers=center, widths=width)(x).degrees.item(),
            stroke_color=INACTIVE_ITEM_2
        )
        return gaussian_graph

    def construct(self):
        method = Text('Wang-Mendel Method', color=BACKGROUND_ITEM)
        self.play(AddTextLetterByLetter(method, run_time=1))
        self.wait(1)
        self.play(FadeOut(method))
        X, env = get_data_and_env()
        self.env_img = display_cart_pole(env, env.state, add_border=False).scale(1).shift(UP*1.1)
        self.fuzzy_sets = {}
        config = load_configuration()
        with config.unfreeze():
            config.fuzzy.partition.kappa = 0.2
            config.fuzzy.partition.epsilon = 0.6
            config.clustering.distance_threshold = 1.0
        linguistic_variables: LinguisticVariables = CLIP(SupervisedDataset(inputs=X, targets=None), config)
        terms: List[Gaussian] = linguistic_variables.inputs
        for term in terms:
            result = term.centers.sort()
            term.centers = torch.nn.Parameter(result.values)
            term.widths = torch.nn.Parameter(term.widths[result.indices])

        labeled_clusters: LabeledClusters = ECM(SupervisedDataset(inputs=X, targets=None), config)
        exemplars = labeled_clusters.clusters.centers.detach().numpy()

        attributes = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']

        linguistic_terms = []
        for idx, term in enumerate(terms):
            if idx == 0:  # cart position
                if term.centers.shape[0] == 2:
                    linguistic_terms.append(['Left', 'Right'])
                elif term.centers.shape[0] == 3:
                    linguistic_terms.append(['Left', 'Middle', 'Right'])
                elif term.centers.shape[0] == 4:
                    linguistic_terms.append(['Left', 'Slightly Left', 'Slightly Right', 'Right'])
            elif idx == 1 or idx == 3:  # cart velocity or pole angle velocity
                if term.centers.shape[0] == 2:
                    linguistic_terms.append(['Low', 'High'])
                elif term.centers.shape[0] == 3:
                    linguistic_terms.append(['Low', 'Moderate', 'High'])
                elif term.centers.shape[0] == 4:
                    linguistic_terms.append(['Very Low', 'Low', 'Moderate', 'High'])
            elif idx == 2:  # pole angle
                if term.centers.shape[0] == 2:
                    linguistic_terms.append(['Left of the Vertical', 'Right of the Vertical'])
                elif term.centers.shape[0] == 3:
                    linguistic_terms.append(['Left of the Vertical', 'Near Zero', 'Right of the Vertical'])
                elif term.centers.shape[0] == 4:
                    linguistic_terms.append(['Slightly Left of the Vertical', 'Left of the Vertical',
                                             'Slightly Right of the Vertical', 'Right of the Vertical'])

        axes = []
        old_axes = None
        for var_idx, variable in enumerate(terms):
            x_min, x_max = X[:, var_idx].min(), X[:, var_idx].max()
            ax = Axes((x_min, x_max), (0, 1), x_length=3, y_length=2, axis_config=dict(
                stroke_color=BACKGROUND_ITEM,
                stroke_width=3,
                numbers_to_exclude=[0],
            ))
            axes.append(ax)
            if old_axes is None:
                ax.to_corner(DL)
            else:
                ax.next_to(old_axes)
            attribute_title = Text(attributes[var_idx], font_size=20, color=BACKGROUND_ITEM)
            attribute_title.next_to(ax, UP)
            # self.play(Create(VGroup(ax, attribute_title)))

            gaussian_graphs = []
            for idx, center in enumerate(variable.centers):
                width = variable.widths[idx]
                if var_idx not in self.fuzzy_sets:
                    self.fuzzy_sets[var_idx] = {'center': [], 'plot': []}
                self.fuzzy_sets[var_idx]['center'].append(center.item())
                gaussian_graph = self.make_fuzzy_set(axes[var_idx], center, width)
                self.fuzzy_sets[var_idx]['plot'].append(gaussian_graph)
                gaussian_graphs.append(gaussian_graph)

            self.play(Create(VGroup(ax, attribute_title, *gaussian_graphs)))

            old_axes = ax
        self.play(FadeIn(self.env_img))

        animated_rules = []
        old_rules, output_rules = set(), set()
        for idx, exemplar in enumerate(exemplars):
            selected_exemplars = exemplars[:idx + 1]
            new_rules: Set[Rule] = wang_mendel_method(
                dataset=SupervisedDataset(inputs=selected_exemplars, targets=None),
                linguistic_variables=linguistic_variables
            )
            diff_rules: List[Rule] = [rule for rule in new_rules if rule not in old_rules]
            if len(diff_rules) == 0:
                continue  # no new rule, skip

            new_rule = diff_rules.pop()
            new_rule = list(new_rule.premise)
            new_rule.sort(key=lambda element: element[0])
            animated_rule = []
            for idx, var_term in enumerate(new_rule):
                var_idx = var_term[0]
                term_idx = var_term[1]
                animated_rule.append((var_term[0], term_idx))

            old_rules.add(frozenset(new_rule))
            animated_rules.append(animated_rule)

            rule = ''
            new_state = []
            animations, areas = [], []
            debug_txt = []
            for var_idx, term_idx in animated_rule:
                debug_txt.append(term_idx)
                rule += attributes[var_idx] + ' is ' + linguistic_terms[var_idx][term_idx]
                if var_idx + 1 < len(animated_rule):
                    rule += ' and\n'
                center = self.fuzzy_sets[var_idx]['center'][term_idx]
                fuzzy_set = self.fuzzy_sets[var_idx]['plot'][term_idx]
                new_center = terms[var_idx].centers[term_idx].item()
                assert center == new_center
                new_state.append(new_center)
                animations.append(fuzzy_set.animate.set_color(ACTIVE_ITEM_2))
                # self.play(fuzzy_set.animate.set_fill('#FFA500', 0.7))
                # calculate area under the curve
                ax = axes[var_idx]
                x_min, x_max = X[:, var_idx].min(), X[:, var_idx].max()
                area = ax.get_area(fuzzy_set, x_range=(x_min, x_max), color=ACTIVE_ITEM_2, opacity=0.4)
                areas.append(area)
                animations.append(FadeIn(area))

            # debug_txt = Text(str(tuple(debug_txt)), color=BACKGROUND_ITEM)
            # self.add(debug_txt)

            new_state = np.array(new_state)
            stacked_state = np.stack([env.state, new_state])  # current state is top row, new state is bottom row
            intermediate_states = []
            for col_idx in range(stacked_state.shape[1]):
                attr_transitions = np.linspace(stacked_state[:, col_idx][0], stacked_state[:, col_idx][1], num=10)
                intermediate_states.append(attr_transitions)
            intermediate_states = np.array(intermediate_states).T
            for intermediate_state in intermediate_states:
                new_env_img = display_cart_pole(env, intermediate_state, add_border=False).scale(1).shift(UP*1.1)
                self.env_img.become(new_env_img)
                self.wait(1e-1)

            new_env_img = display_cart_pole(env, new_state, add_border=False).scale(1).shift(UP*1.1)
            self.env_img.become(new_env_img)
            self.wait(1e-1)

            text_rule = Text(rule, font_size=24, color=BACKGROUND_ITEM)
            text_rule.to_edge(UP)
            animations.append(AddTextLetterByLetter(text_rule, run_time=1))
            self.play(*animations)
            self.wait(2)
            animations = []
            for var_idx, term_idx in animated_rule:
                animations.append(self.fuzzy_sets[var_idx]['plot'][term_idx].animate.set_color(INACTIVE_ITEM_2))
                animations.append(FadeOut(areas[var_idx]))
            animations.append(RemoveTextLetterByLetter(text_rule, run_time=1))
            self.play(*animations)
            print(rule)
            # self.remove(debug_txt)
        self.wait(1)


if __name__ == '__main__':
    c = WMDemo()
    c.render()
