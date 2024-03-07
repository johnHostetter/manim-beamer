import numpy as np
import torch

from manim import *
from colors import *
from time import time
from copy import deepcopy
from sklearn import manifold
from sklearn.manifold import TSNE

from soft.datasets import SupervisedDataset
from soft.utilities.reproducibility import set_rng, load_configuration
from soft.fuzzy.unsupervised.cluster.online.ecm import apply_evolving_clustering_method as ECM, \
    LabeledClusters
from common import make_axes, add_labels_to_axes, display_cart_pole, get_data_and_env

set_rng(2)


def extract_sequence(tsne, X):
    sklearn_grad = manifold._t_sne._gradient_descent
    Y_seq = []

    # modified from sklearn source
    # https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/manifold/t_sne.py#L442
    # to save the sequence of embeddings at each training iteration
    def _gradient_descent(objective, p0, it, n_iter,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error: np.float64 = np.finfo(float).max
        best_error: np.float64 = np.finfo(float).max
        best_iter = i = it

        tic = time()
        for i in range(it, n_iter):

            # save the current state
            Y_seq.append(p.copy().reshape(-1, 2))

            error, grad = objective(p, *args, **kwargs)
            grad_norm = np.linalg.norm(grad)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if (i + 1) % n_iter_check == 0:
                toc = time()
                duration = toc - tic
                tic = toc

                if verbose >= 2:
                    print("[t-SNE] Iteration %d: error = %.7f,"
                          " gradient norm = %.7f"
                          " (%s iterations in %0.3fs)"
                          % (i + 1, error, grad_norm, n_iter_check, duration))

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: did not make any progress "
                              "during the last %d episodes. Finished."
                              % (i + 1, n_iter_without_progress))
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                              % (i + 1, grad_norm))
                    break

        return p, error, i

    # replace with modified gradient descent
    manifold._t_sne._gradient_descent = _gradient_descent
    # train given tsne object with new gradient function
    X_proj = tsne.fit_transform(X)
    # return to default version
    manifold._t_sne._gradient_descent = sklearn_grad

    return np.array(Y_seq)


def run_ecm(scene, axes, tsne_X, X, visited_X, env):
    animations, successions, rewind_animations = [], [], []
    visited_clusters = set()
    old_clusters_supports = np.array([0.])
    for iter, tsne_x in enumerate(tsne_X):  # get the last tsne X
        tsne_X = (tsne_X - tsne_X.min(0)[None, :]) / (tsne_X.max(0)[None, :] - tsne_X.min(0)[None, :])

        config = load_configuration()
        with config.unfreeze():
            config.clustering.distance_threshold = 0.4
        labeled_clusters: LabeledClusters = ECM(
            SupervisedDataset(inputs=torch.tensor(tsne_X[:iter + 1]), targets=None),
            config=load_configuration()
        )
        new_clusters_supports = np.array(labeled_clusters.supports)
        dot = scene.data_dots[iter]
        if len(new_clusters_supports) > len(old_clusters_supports):  # a new cluster identified with x
            cluster_idx = len(new_clusters_supports) - 1
        else:
            cluster_idx = np.argmax(new_clusters_supports - old_clusters_supports)
        previous_spot_dot = deepcopy(dot)
        scene.add(previous_spot_dot)
        animations.append(previous_spot_dot.animate.set_opacity(0.25))
        center = labeled_clusters.clusters.centers[cluster_idx].detach().numpy()
        circle = Circle(radius=config.clustering.distance_threshold)
        circle.set_stroke(ACTIVE_ITEM_2, 3)
        circle.move_to(axes.c2p(center[0], center[1]))
        animations.append(dot.animate.move_to(axes.c2p(center[0], center[1])))
        animations.append(Create(circle))

        # display the exemplar
        print(iter)
        if tuple(center) not in visited_clusters:
            state = X[iter].detach().numpy()
            if tuple(state) not in visited_X:
                cart_pole_img = display_cart_pole(env, state).scale(1e-2)
                cart_pole_img.move_to(axes.c2p(center[0], center[1]))
                successions.append(Succession(
                    FadeIn(cart_pole_img), ScaleInPlace(cart_pole_img, 50), Wait(run_time=1),
                    ScaleInPlace(cart_pole_img, 1e-2), FadeOut(cart_pole_img)))
                visited_X.add(tuple(state))
                break  # exit for loop

        visited_clusters.add(tuple(center))

        rewind_animations.append(Uncreate(circle))
        old_clusters_supports = new_clusters_supports
    scene.play(*animations)
    for succession in successions:
        scene.play(succession)
    return rewind_animations


class ECMDemo(Scene):
    def __init__(self, **kwargs):
        super().__init__()
        background = ImageMobject('background.png').scale(2).set_color('#FFFFFF')
        self.add(background)

    def construct(self):
        method = Text('Evolving Clustering Method', color=BACKGROUND_ITEM)
        self.play(AddTextLetterByLetter(method, run_time=1))
        self.wait(1)
        self.fuzzy_sets, self.data_dots = [], []
        X, env = get_data_and_env()
        X = X[:1000]
        tsne = TSNE(n_iter=300, perplexity=5, verbose=True)
        tsne._EXPLORATION_N_ITER = 300
        Y_seq = extract_sequence(tsne, X)

        axes = make_axes(self, min_x=0, max_x=1, step_x=0.1, min_y=0, max_y=1, step_y=0.1)
        x_axis_lbl, y_axis_lbl = add_labels_to_axes(axes, x_label='t-SNE 1', y_label='t-SNE 2')
        self.play(RemoveTextLetterByLetter(method, run_time=1), Create(VGroup(axes, x_axis_lbl, y_axis_lbl)))

        visited_X = set()
        for tsne_iter, tsne_X in enumerate(Y_seq[10:]):
            if tsne_iter % 60 == 0:
                animations = []
                tsne_X = (tsne_X - tsne_X.min(0)[None, :]) / (tsne_X.max(0)[None, :] - tsne_X.min(0)[None, :])
                for idx, tsne_x in enumerate(tsne_X):
                    try:
                        dot = self.data_dots[idx]
                        animations.append(dot.animate.move_to(axes.c2p(tsne_x[0], tsne_x[1])))
                    except IndexError:
                        dot = Dot(color=ACTIVE_ITEM_1)
                        self.data_dots.append(dot)
                        dot.move_to(axes.c2p(tsne_x[0], tsne_x[1]))
                        animations.append(FadeIn(dot))
                self.wait(3)
                self.play(*animations)

                rewind = run_ecm(self, axes, tsne_X, X, visited_X, env)
                self.wait(3)
                if len(rewind) > 0:
                    self.play(*rewind)  # undo the animations from ECM
                break


if __name__ == '__main__':
    c = ECMDemo()
    c.render()
    # c.construct()
