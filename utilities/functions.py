"""
Utility functions, such as for getting the powerset of an iterable.
"""
import inspect
from typing import Dict, Any
from collections.abc import Iterable
from itertools import chain, combinations

import torch
import numpy as np


def powerset(iterable: Iterable, min_items: int):
    """
    Get the powerset of an iterable.

    Args:
        iterable: An iterable collection of elements.
        min_items: The minimum number of items that must be in each subset.

    Returns:
        The powerset of the given iterable.
    """
    # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    return chain.from_iterable(
        combinations(list(iterable), r)
        for r in range(min_items, len(list(iterable)) + 1)
    )


def convert_to_tensor(values: np.ndarray) -> torch.Tensor:
    """
    If the given values are not torch.Tensor, convert them to torch.Tensor.

    Args:
        values: Values such as the centers or widths of a fuzzy set.

    Returns:
        torch.tensor(np.array(values))
    """
    if isinstance(values, torch.Tensor):
        return values
    return torch.tensor(np.array(values)).float()


def get_object_attributes(obj_instance) -> Dict[str, Any]:
    # get the attributes that are local to the class, but may be inherited from the super class
    local_attributes = inspect.getmembers(
        obj_instance,
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are inherited from (or found within) the super class
    super_attributes = inspect.getmembers(
        obj_instance.__class__.__bases__[0],
        lambda attr: not (inspect.ismethod(attr)) and not (inspect.isfunction(attr)),
    )
    # get the attributes that are local to the class, but not inherited from the super class
    return {
        attr: value
        for attr, value in local_attributes
        if (attr, value) not in super_attributes and not attr.startswith("_")
    }


class GaussianDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception("p value should accomplish 0 < p < 1")
        self.p = p

    def forward(self, x):
        if self.training:
            standard_deviation = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.rand_like(x) * standard_deviation
            return x * epsilon
        else:
            return x


def raw_dropout(x, p):
    # generate a binary mask based on the dropout probability
    s = list(x.shape)
    s[-1] = 2

    weights = torch.empty(s, dtype=torch.float)
    weights[:, :, 0] = p
    weights[:, :, 1] = 1 - p
    mask = torch.multinomial(
        weights.view(-1, 2),
        num_samples=x.shape[-1],
        replacement=True,
    ).view(x.shape)

    # apply the mask to the input tensor
    return x * mask  # my defn, weight balancing
    # return (x * mask) / (1 - p)
