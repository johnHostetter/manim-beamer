import gym
import numpy as np

from torch.utils.data import Dataset
from d3rlpy.dataset import MDPDataset

"""
Create a class that will assist in quickly loading the data for offline training.
"""

from fql import FQLModel
from fis import mountain_car_fis


class Agent:
    def __init__(self, model):
        self.model = model

    def get_initial_action(self, state):
        try:
            return self.model.get_initial_action(state)
        except AttributeError:
            return self.model.get_action(state)

    def get_action(self, state):
        return self.model.get_action(state)

    def learn(self, state, reward, trajectories):
        return self.model.run(state, reward)


def mountain_car():
    # --- online & static Fuzzy Q-Learning ---

    fis = mountain_car_fis()

    model = FQLModel(gamma=0.99,
                     alpha=0.1,
                     ee_rate=1.,
                     action_set_length=3,
                     fis=fis)

    parameters = {
        'environment': 'MountainCar-v0',
        'seed': 0, 'max episodes': 100,
        'agent': Agent(model), 'verbose': True, 'render': False
    }
    game = Game(parameters)

    print('Observation shape:', game.env.observation_space.shape)
    print('Action length:', game.env.action_space.n)
    action_set_length = game.env.action_space.n

    return game, game.play(True, False)


class Game:
    """
    This wrapper will work for the following OpenAI gyms:
        MountainCar-v0, CartPole-v1, LunarLander-v2

    If agent is None, then random play will occur;
    else, the agent will be called with: get_action(state).
    """

    def __init__(self, parameters):  # None agent is random play
        self.environment_name = parameters['environment']
        self.env = gym.make(parameters['environment'])
        self.env = self.env.unwrapped
        self.env.seed(parameters['seed'])
        self.env.action_space.seed(parameters['seed'])
        self.max_episodes = parameters['max episodes']
        self.agent = parameters['agent']
        self.verbose = parameters['verbose']
        self.render = parameters['render']

        if self.verbose:
            print('Observation shape:', self.env.observation_space.shape)
            print('Action length:', self.env.action_space.n)

    def get_agent_action(self, state, initial=False):
        if self.agent is None:
            return self.env.action_space.sample()
        else:
            if initial:
                return self.agent.get_initial_action(state)
            else:
                return self.agent.get_action(state)

    def play(self, agent_is_training, exploit=False):  # agent_is_training is a boolean, True changes agent's parameters
        done = True  # whether the current episode is done
        iteration = 0
        # trajectories = []  # a list of all tuples in the form of (s, a, r, s', done)
        data = {'observation': [], 'action': [], 'reward': [], 'terminal': []}
        continue_loop = iteration < self.max_episodes
        while continue_loop:
            if done:
                state_value = self.env.reset()
                action = self.get_agent_action(state_value)  # the initial action for an episode

                iteration += 1
                r = 0

                if self.agent is not None:
                    # epsilon decay
                    try:
                        self.agent.model.ee_rate -= self.agent.model.ee_rate * 0.01

                        if not exploit and self.agent.model.ee_rate <= 0.2:
                            self.agent.model.ee_rate = 0.2
                    except AttributeError:
                        # no ee rate given, assuming greedy policy
                        self.agent.model.ee_rate = 0.0

            # render the environment for the last couple episodes
            if self.render and iteration + 1 > (self.max_episodes - 6):
                self.env.render()

            prev_state = state_value
            state_value, reward, done, _ = self.env.step(action)

            if agent_is_training:
                # learn from this and get the next action
                action = self.agent.learn(state_value, reward, data)
            else:
                # get the next action
                action = self.get_agent_action(state_value)

            r += reward

            # reach 2000 steps --> done
            if self.environment_name == 'MountainCar-v0' and r <= -2000:
                done = True

            data['observation'].append(prev_state)
            data['action'].append(action)
            data['reward'].append(reward)
            data['terminal'].append(done)

            if self.verbose:
                if done:
                    print('Episode: {}; Total Reward: {}'.format(iteration + 1, r))
                try:  # if the agent has an epsilon parameter, report its current value
                    pass
                    # print('Epsilon=', self.agent.model.ee_rate)
                except AttributeError:
                    pass  # no epsilon value to report since the agent is not being trained online

            continue_loop = iteration < self.max_episodes  # whether to keep collecting data

        self.env.close()

        # the observations and actions must be 2-dimensional Numpy arrays
        # the rewards and terminals can be 1-dimensional Numpy arrays
        return MDPDataset(
            np.array(data['observation']), np.array(data['action'])[..., None],
            np.array(data['reward']), np.array(data['terminal']).astype(int)
        )


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
