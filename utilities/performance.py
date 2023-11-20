"""
Implement functions that may be helpful in improving performance, such as enabling/disabling
certain features when the debugger is active.
"""
import sys
from typing import Any, Callable

import gym
import torch
import numpy as np
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.preprocessing.stack import StackedObservation


# https://stackoverflow.com/questions/38634988/check-if-program-runs-in-debug-mode
def is_debugger_active() -> bool:
    """
    Determine if the debugger is currently active.

    Returns:
        True or False
    """
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def performance_boost() -> bool:
    """
    If the debugger is not active, improve the performance of Torch by disabling some features.

    Returns:
        True if the performance was boosted, False otherwise.
    """
    # for performance boost, disable the following when we are not going to debug:
    toggle_performance_boost = not is_debugger_active()
    if toggle_performance_boost:
        torch.autograd.set_detect_anomaly(is_debugger_active())
        torch.autograd.profiler.profile(is_debugger_active())
        torch.autograd.profiler.emit_nvtx(is_debugger_active())
    return toggle_performance_boost


def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
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
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        algo.impl._q_func.q_funcs[0]._encoder.flc.eval()

        if is_image:
            stacked_observation = StackedObservation(observation_shape, algo.n_frames)

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        with torch.no_grad():
                            action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        with torch.no_grad():
                            action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        algo.impl._q_func.q_funcs[0]._encoder.flc.train()
        return float(np.mean(episode_rewards))

    return scorer
