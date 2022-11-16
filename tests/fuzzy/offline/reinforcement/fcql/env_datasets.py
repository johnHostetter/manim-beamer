import gym
import numpy as np

from torch.utils.data import Dataset
from d3rlpy.dataset import MDPDataset

"""
Create a class that will assist in quickly loading the data for offline training.
"""


class CartPoleDataset(Dataset):
    """
    Offline reinforcement learning dataset of Cart Pole
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision
    """

    def __init__(self, data):
        self.dataset = data
        self.transitions, self.unique_states = self.transform_data()

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]

    def transform_data(self):
        states = []
        transitions = []
        for episode in self.dataset:
            for transition in episode.transitions:
                done = transition.terminal == 1.0
                states.append(list(transition.observation))
                value = {'state': transition.observation, 'action': transition.action, 'reward': transition.reward,
                         'next state': transition.next_observation, 'terminal': done}
                transitions.append(value)
        return transitions, np.array(states)
