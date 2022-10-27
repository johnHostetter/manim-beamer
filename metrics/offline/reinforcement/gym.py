import torch
import numpy as np


class SummaryStatistics:
    def __init__(self, values):
        self.values = values

    def min(self):
        return np.min(self.values)

    def mean(self):
        return np.mean(self.values)

    def std(self):
        return np.std(self.values)

    def median(self):
        return np.median(self.values)

    def max(self):
        return np.max(self.values)


"""
The following code block comes from the `d3rlpy` library, 
but it has been modified to return both the average and 
standard deviation of the online evaluation.
"""


def evaluate_on_environment(env, n_trials=100, epsilon=0.0, render=False):
    """ Returns scorer function of evaluation on environment.

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
        render (bool): flag to render environment.

    Returns:
        callable: scorer function.


    """

    def scorer(algo, *args):
        print('Evaluating online for {} episodes.'.format(n_trials))
        episode_rewards = []
        for trial_idx in range(n_trials):
            observation, info = env.reset()
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = torch.argmax(algo.predict(torch.tensor(np.array([observation])))).item()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

                if terminated:
                    break
            print('Episode {}: {}'.format(trial_idx + 1, episode_reward))
            episode_rewards.append(episode_reward)
        return SummaryStatistics(episode_rewards)

    return scorer