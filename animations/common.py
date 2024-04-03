from typing import Tuple
from dataclasses import dataclass

import cv2  # pip install opencv-python
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from manim import *
from d3rlpy.datasets import get_cartpole

from experiments.reinforcement.common import CustomDataset


@dataclass
class ItemColor:
    ACTIVE_1: str = "#FD56DC"  # hot pink
    INACTIVE_1: str = "#FFB9CB"  # light pink
    ACTIVE_2: str = "#68EF00"  # hot pink
    INACTIVE_2: str = "#01D3FC"  # light pink
    BACKGROUND: str = "#025393"  # dark blue


class AxisConfig:
    def __init__(self, min_value: float, max_value: float, step: float, length: float = 10.0):
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.length = length

    def get_range(self) -> Tuple[float, float, float]:
        return self.min_value, self.max_value, self.step


def make_axes(
    scene,
    x_axis_config: AxisConfig,
    y_axis_config: AxisConfig,
    stroke_width=3,
    axes_color=ItemColor.BACKGROUND,
):
    axes = Axes(
        x_range=x_axis_config.get_range(),
        y_range=y_axis_config.get_range(),
        x_length=x_axis_config.length,
        y_length=y_axis_config.length,
        # The axes will be stretched to match the specified
        # height and width
        # Axes is made of two NumberLine objects.  You can specify
        # their configuration with axis_config
        axis_config=dict(
            tip_shape=StealthTip,
            stroke_color=axes_color,
            stroke_width=stroke_width,
            numbers_to_exclude=[0],
            include_numbers=True,
            decimal_number_config=dict(color=axes_color),
        ),
        # # Alternatively, you can specify configuration for just one
        # # of them, like this.
        # y_axis_config=dict(
        #     numbers_with_elongated_ticks=[-2, 2],
        # )
    )

    # scene.add(axes)
    return axes


def add_labels_to_axes(ax, x_label, y_label, text_color=ItemColor.BACKGROUND):
    x_axis_lbl = Text(x_label, font_size=24, color=str(text_color))
    y_axis_lbl = Text(y_label, font_size=24, color=str(text_color))

    x_axis_lbl.next_to(ax, DOWN)
    y_axis_lbl.rotate(1.5708)
    y_axis_lbl.next_to(ax, LEFT)
    return x_axis_lbl, y_axis_lbl


def get_data_and_env(n_samples=100):
    data, _ = get_cartpole(dataset_type="replay")
    X = torch.tensor(np.vstack(([episode.observations for episode in data.episodes])))
    X = X[np.lexsort((X[:, 0], X[:, 2]), axis=0)]
    idx = np.round(np.linspace(0, len(X) - 1, 100)).astype(int)
    X = X[idx]
    np.random.shuffle(X)
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = env.unwrapped
    env.reset()
    return X[:n_samples], env


def display_cart_pole(env, state, add_border=True):
    env.state = state
    img = plt.imshow(env.render())
    plt.grid(False)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("media/images/cartpole.png", dpi=100)

    if add_border:  # add a black border to the image
        img = cv2.imread("media/images/cartpole.png")
        img = cv2.copyMakeBorder(
            src=img,
            top=15,
            bottom=15,
            left=15,
            right=15,
            borderType=cv2.BORDER_CONSTANT,
        )
        cv2.imwrite("media/images/cartpole.png", img)

    return ImageMobject("media/images/cartpole.png")
