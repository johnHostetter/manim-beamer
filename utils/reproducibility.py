"""
Provides functions that help guarantee reproducibility.
"""
import os
import random

import torch
import numpy as np


def set_rng(seed):
    """
    Set the random number generator.

    Args:
        seed:

    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def env_seed(env, seed):
    """
    Set the random number generator, and also set the random number generator for gym.env.

    Args:
        env:
        seed:

    Returns:
        None
    """
    set_rng(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
