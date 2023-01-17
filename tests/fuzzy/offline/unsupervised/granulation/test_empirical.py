import os
import torch
import pathlib
import unittest
import numpy as np

from soft.fuzzy.sets import Gaussian
from sklearn import datasets
from utils.reproducibility import set_rng

set_rng(0)


def example():
    return torch.tensor([[1., 2., 3.],
                         [1., 2., 3.],
                         [2., 3., 4.],
                         [4., 1., 5.],
                         [7., 9., 5.],
                         [6., 12., 4.],
                         [3., 2., 1.],
                         [8., 4., 2.]])


def multimodal_density(X):
    unique_observations, frequencies = X.unique(dim=0, return_counts=True)
    distance = torch.cdist(unique_observations, unique_observations)
    numerator = torch.pow(distance, 2).sum()
    denominator = 2 * X.shape[0] * distance.sum(dim=-1)
    return frequencies * (numerator / denominator)


class TestEDA(unittest.TestCase):
    def test_frequencies(self):
        X = example()
        unique_observations, frequencies = np.unique(X, axis=0, return_counts=True)

        expected_observations = np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 2, 1],
            [4, 1, 5],
            [6, 12, 4],
            [7, 9, 5],
            [8, 4, 2]
        ])
        expected_frequencies = np.array([2, 1, 1, 1, 1, 1, 1])
        assert np.isclose(unique_observations, expected_observations).all()
        assert np.isclose(frequencies, expected_frequencies).all()

    def test_euclidean_distance(self):
        A = torch.rand((5, 2))  # 5 rows of 2 column vectors
        B = torch.rand((5, 2))  # 5 rows of 2 column vectors
        distance = torch.pow(A - B, 2).sum(dim=-1).sqrt()
        expected_distance = torch.tensor(
            [0.3950208, 0.07571248, 0.11634678, 0.22888084, 0.4575092]
        )
        assert torch.isclose(distance, expected_distance).all()

    def test_multimodal_density(self):
        X = example()

    def test_unimodal_density(self):
        X = example()
        distance = torch.cdist(X, X)
        numerator = torch.pow(distance, 2).sum()
        denominator = 2 * X.shape[0] * distance.sum(dim=-1)
        densities = numerator / denominator

    def test_multimodal_density(self):
        X = example()
        unique_observations, frequencies = X.unique(dim=0, return_counts=True)
        distance = torch.cdist(unique_observations, unique_observations)
        numerator = torch.pow(distance, 2).sum()
        denominator = 2 * X.shape[0] * distance.sum(dim=-1)
        densities = frequencies * (numerator / denominator)

        # finding the maximum MM density
        visited_indices = []
        cluster = set()
        index = densities.max(dim=0).indices.item()
        visited_indices.append(index)
        selected = unique_observations[index]
        cluster.add(selected)
        distance.fill_diagonal_(float('inf'))
        while len(visited_indices) < unique_observations.shape[0]:
            temp = distance[index]
            temp[torch.LongTensor(visited_indices)] = float('inf')
            value, index = temp.min(dim=0)
            visited_indices.append(index.item())
            cluster.add(unique_observations[index.item()])
            # sort_indices = distance[index].sort().indices
        print(visited_indices)
        local_maxima = densities[visited_indices]
        print(densities[visited_indices])

        peak_mask = torch.cat([torch.zeros(1, dtype=torch.bool), (local_maxima[:-2] < local_maxima[1:-1])
                               & (local_maxima[2:] < local_maxima[1:-1]),
                               torch.zeros(1, dtype=torch.bool)], dim=0)
        peak_mask[0] = local_maxima[0] > local_maxima[1]
        peak_mask[-1] = local_maxima[-1] > local_maxima[-2]
        prototypes = unique_observations[peak_mask]
        distances = torch.cdist(unique_observations, prototypes)
        cloud_labels = distances.min(dim=1).indices
        cloud_centers = []
        for label in cloud_labels.unique():
            cloud_data = unique_observations[cloud_labels == label]
            cloud_centers.append(cloud_data.mean(dim=0).detach().numpy())
        cloud_centers = torch.tensor(np.array(cloud_centers))
        cloud_distances = torch.cdist(prototypes, prototypes)
        eta = cloud_distances.mean()
        sigma = cloud_distances.std()
        R = sigma * (1 - sigma / eta)
        densities = multimodal_density(cloud_centers)
