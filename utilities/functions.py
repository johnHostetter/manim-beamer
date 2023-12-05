"""
Utility functions, such as for getting the powerset of an iterable.
"""
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
