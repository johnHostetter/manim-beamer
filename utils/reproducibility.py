"""
Provides functions that help guarantee reproducibility.
"""
import os
import random
import pathlib
from typing import Union

import torch
import numpy as np

from YACS.yacs import Config


def set_rng(seed: int):
    """
    Set the random number generator.

    Args:
        seed:

    Returns:
        None
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def env_seed(env, seed: int):
    """
    Set the random number generator, and also set the random number generator for gym.env.

    Args:
        env:
        seed:

    Returns:
        None
    """
    set_rng(seed)
    try:
        env.reset(seed=seed)
    except TypeError:  # older version of gym (e.g., 0.21)
        env.seed(seed)
    env.action_space.seed(seed)


def load_configuration(
    file_name: Union[str, pathlib.Path] = "default_config.yaml"
) -> Config:
    """
    Load and return the default configuration that should be used for models, if another
    overriding configuration is not used in its place.

    Args:
        file_name: Union[str, pathlib.Path] Either a file name (str) where the function will look up
        the *.yml configuration file on the parent directory (i.e., git repository) level, or a
        pathlib.Path where the object redirects the function to a specific location that may be in
        a subdirectory of this repository.

    Returns:
        YACS.yacs.Config
    """
    file_path = pathlib.Path(__file__).parent.parent / file_name
    return Config(str(file_path))
