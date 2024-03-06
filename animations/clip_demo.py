from manim import *
from colors import *
from soft.datasets import SupervisedDataset
from soft.fuzzy.logic.rules import LinguisticVariables
from soft.utilities.reproducibility import set_rng, load_configuration
from common import make_axes, add_labels_to_axes, get_data_and_env
from soft.fuzzy.sets.continuous.impl import Gaussian
from soft.fuzzy.unsupervised.granulation.online.clip import (
    apply_categorical_learning_induced_partitioning as CLIP
)

set_rng(1)


class CLIPDemo(Scene):
    def __init__(self, **kwargs):
        super().__init__()
        background = ImageMobject('background.png').scale(2).set_color('#FFFFFF')
        self.add(background)
        self.fuzzy_sets = None

    def add_fuzzy_set(self, axes, center, width, x: float, dot, new_terms):
        temp_gaussian: Gaussian = Gaussian(centers=center, widths=width)
        gaussian_graph = axes.plot(
            lambda x: temp_gaussian(x).degrees.item(),
            stroke_color=ACTIVE_ITEM_2
            # use_smoothing=True,
            # color=ORANGE
        )
        gaussian_label = axes.get_graph_label(gaussian_graph, Text('New Fuzzy Set'), color=ACTIVE_ITEM_2, direction=UP)
        self.fuzzy_sets.append(gaussian_graph)
        # self.add(gaussian_graph)
        self.play(
            Create(gaussian_graph),
            FadeIn(gaussian_label),
            dot.animate.move_to(axes.c2p(x, new_terms(x).degrees.max().item()))
        )
        self.wait()
        self.play(
            FadeOut(gaussian_label),
            gaussian_graph.animate.set_color(INACTIVE_ITEM_2),
            dot.animate.set_color(INACTIVE_ITEM_1),
            # dot.animate.set_glow_factor(1.0)
        )

    def revise_fuzzy_sets(self, axes, new_terms, X):
        if new_terms is not None:
            animations = []
            for idx, center in enumerate(new_terms.centers.flatten()):
                gaussian_graph = axes.plot(
                    lambda x: new_terms(x).degrees[idx].detach().numpy().item(),
                    stroke_color=INACTIVE_ITEM_2
                    # use_smoothing=True,
                    # color=GREEN
                )
                try:
                    animations.append(
                        self.fuzzy_sets[idx].animate.become(gaussian_graph))
                    animations.extend(self.revise_data_points(axes, new_terms, X))
                except IndexError:  # there is no fuzzy set located at 'idx'
                    continue
            if len(animations) > 0:
                self.play(*animations)

    def revise_data_points(self, axes, new_terms, X):
        animations = []
        if self.data_dots is not None:
            for idx, dot in enumerate(self.data_dots):
                x = X[idx]
                animations.append(
                    dot.animate.move_to(axes.c2p(x.flatten().item(), new_terms(x).degrees.max().item())),
                )
        return animations

    def construct(self):
        method = Text('Categorical Learning-Induced Partitioning', color=BACKGROUND_ITEM)
        self.play(AddTextLetterByLetter(method, run_time=1))
        self.wait(1)
        X, env = get_data_and_env(n_samples=10)
        X = X[:, :1]
        config = {
            'minimums': X.min(0).values,
            'maximums': X.max(0).values,
            'eps': 0.05,
            'kappa': 0.6
        }
        self.fuzzy_sets, self.data_dots = [], []
        axes = make_axes(self, min_x=X.min(), max_x=X.max(), step_x=0.25, min_y=0, max_y=1, step_y=0.1)
        x_axis_lbl, y_axis_lbl = add_labels_to_axes(axes, x_label='Cart Position',
                                                    y_label='Degree of Membership')

        self.play(RemoveTextLetterByLetter(method, run_time=1), Create(VGroup(axes, x_axis_lbl, y_axis_lbl)))

        old_terms, new_terms = None, None
        for idx, x in enumerate(X):
            x: float = x.item()  # x is a 1D tensor
            dot = Dot(color=ACTIVE_ITEM_1)
            self.data_dots.append(dot)
            dot.move_to(axes.c2p(0, 0))
            self.play(dot.animate.move_to(axes.c2p(x, 0)))

            if old_terms is not None:
                degree = old_terms(x).degrees.max().item()
                self.play(dot.animate.move_to(axes.c2p(x, degree)))
                line_graph = axes.plot(
                    lambda x: config['kappa'],
                    stroke_color=ACTIVE_ITEM_2
                )
                dashed_line_graph = DashedVMobject(line_graph)
                self.play(
                    Create(dashed_line_graph), run_time=2)
                self.wait()
                if degree >= config['kappa']:
                    message = Text('Satisfied')
                else:
                    message = Text('Not Satisfied')
                dashed_line_label = axes.get_graph_label(line_graph, message, color=ACTIVE_ITEM_2, direction=UP)
                self.play(FadeIn(dashed_line_label))
                self.wait()
                self.play(FadeOut(dashed_line_label))
                self.play(
                    FadeOut(dashed_line_graph)
                )

            selected_X = X[:idx + 1]
            if selected_X.ndim == 1:
                selected_X = selected_X.unsqueeze(dim=1)
            linguistic_variables: LinguisticVariables = CLIP(
                dataset=SupervisedDataset(inputs=selected_X, targets=None),
                config=load_configuration()
            )
            new_terms = linguistic_variables.inputs[0]
            self.revise_fuzzy_sets(axes, new_terms, X)

            if old_terms is None or new_terms.centers.flatten().shape[0] > old_terms.centers.flatten().shape[0]:
                # new fuzzy set
                if new_terms.centers.ndim == 0:
                    center, width = new_terms.centers.item(), new_terms.widths.item()
                else:
                    center, width = new_terms.centers[-1].item(), new_terms.widths[-1].item()

                self.add_fuzzy_set(axes, center, width, x, dot, new_terms)

            else:
                self.play(
                    dot.animate.move_to(axes.c2p(x, new_terms(x).degrees.max().item())),
                    # dot.animate.set_color(PURPLE_A),
                    dot.animate.set_glow_factor(1.0)
                )
            # self.revise_fuzzy_sets(axes, new_terms, X)
            self.play(dot.animate.set_color(INACTIVE_ITEM_1))
            self.wait()
            old_terms = new_terms
        self.wait(1)


if __name__ == '__main__':
    c = CLIPDemo()
    c.render()
    # c.construct()

