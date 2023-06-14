"""
Implement SummaryStatistics class and a function 'evaluate_on_environment' where the evaluation
on the environment returns an instance of SummaryStatistics - an object that allows for various
metrics to be easily applied.
"""
from collections import deque

import gym
import torch
import numpy as np


class SummaryStatistics:
    """
    A class that allows easy access to various metrics of interest.
    """

    def __init__(self, values):
        self.values = values
        self.params = {}

    def min(self):
        """
        Calculate the minimum of the scores.

        Returns:
            Minimum of self.values.
        """
        return np.min(self.values)

    def mean(self):
        """
        Calculate the mean of the scores.

        Returns:
            Mean of self.values.
        """
        return np.mean(self.values)

    def std(self):
        """
        Calculate the standard deviation of the scores.

        Returns:
            Standard deviation of self.values.
        """
        return np.std(self.values)

    def median(self):
        """
        Calculate the median of the scores.

        Returns:
            Median of self.values.
        """
        return np.median(self.values)

    def max(self):
        """
        Calculate the maximum of the scores.

        Returns:
            Maximum of self.values.
        """
        return np.max(self.values)

    def to_dict(self):
        """
        Store metric results in dictionary.

        Returns:
            A dictionary.
        """
        return {
            "min": self.min(),
            "mean": self.mean(),
            "std": self.std(),
            "median": self.median(),
            "max": self.max(),
        }


# The following code block comes from the `d3rlpy` library,
# but it has been modified to return both the average and
# standard deviation of the online evaluation.


def evaluate_on_environment(env, n_trials=100, epsilon=0.0, text=True, render=False):
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env (gym.Env): gym-styled environment.
        n_trials (int): the number of trials.
        epsilon (float): noise factor for epsilon-greedy policy.
        text (bool): flag to render text output.
        render (bool): flag to render environment.

    Returns:
        callable: scorer function.


    """

    def scorer(algo, *args):
        print(f"Evaluating online for {n_trials} episodes with: {args}.")
        scores_window = deque(maxlen=100)  # last 100 scores
        for trial_idx in range(n_trials):
            if gym.__version__ <= "0.21.0":
                observation = env.reset()  # observation only
            else:
                observation, _ = env.reset()  # environment, info
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = torch.argmax(
                        algo.predict(torch.tensor(np.array([observation])))
                    ).item()
                if gym.__version__ <= "0.21.0":
                    observation, reward, terminated, _ = env.step(action)
                    truncated = False
                else:
                    observation, reward, terminated, truncated, _ = env.step(
                        action
                    )  # last is 'info'
                episode_reward += reward

                if render:
                    env.render()

                if terminated or truncated:
                    break

            if text and len(scores_window) > 0:
                print(
                    f"\rEpisode: {trial_idx + 1}\tAverage Score: {np.mean(scores_window):.6f}",
                    end="",
                )
                if trial_idx > 0 and trial_idx % 100 == 0:
                    print(
                        f"\rEpisode: {trial_idx + 1}\tAverage Score: {np.mean(scores_window):.6f}"
                    )

            scores_window.append(episode_reward)
        return SummaryStatistics(scores_window)

    return scorer
