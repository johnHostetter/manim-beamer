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
