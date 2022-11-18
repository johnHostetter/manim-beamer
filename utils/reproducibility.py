import os
import torch
import random
import numpy as np


def set_rng(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def env_seed(env, seed):
    set_rng(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
