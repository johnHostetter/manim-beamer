#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 22:56:26 2021

@author: john

    This code demonstrates the Fuzzy Rule-Based Q-Learning Algorithm.
    It is corrected from the FQL code at the following link:
        https://github.com/seyedsaeidmasoumzadeh/Fuzzy-Q-Learning
"""

import copy
import random
import operator
import itertools
import functools
import numpy as np

from fis import Build

GLOBAL_SEED = 0
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


class FQLModel(object):
    def __init__(self, gamma, alpha, ee_rate,
                 action_set_length, fis=Build()):
        self.R = []
        self.R_ = []
        self.M = []
        self.V = []
        self.Q = [0]
        self.Error = 0
        self.gamma = gamma
        self.alpha = alpha
        self.ee_rate = ee_rate
        self.action_set_length = action_set_length
        self.fis = fis

        self.q_table = np.zeros((self.fis.get_number_of_rules(),
                                 self.action_set_length))

    # Fuzzify to get the degree of truth values
    def truth_value(self, state_value):
        self.R = []
        L = []
        input_variables = self.fis.list_of_input_variable
        for index, variable in enumerate(input_variables):
            m_values = []
            fuzzy_sets = variable.get_fuzzy_sets()
            for fuzzy_set in fuzzy_sets:
                membership_value = fuzzy_set.membership_value(state_value[index])
                m_values.append(membership_value)
            L.append(m_values)

        # Calculate Truth Values
        # results are the product of membership functions
        for element in itertools.product(*L):
            self.R.append(functools.reduce(operator.mul, element, 1))
        return self

    def action_selection(self):
        self.M = []
        r = random.uniform(0, 1)

        for rule in self.q_table:
            # Act randomly
            if r < self.ee_rate:
                action_index = random.randint(0, self.action_set_length - 1)
            # Get maximum values
            else:
                action_index = np.argmax(rule)
            self.M.append(action_index)

        # 1. Action = sum of truth values*action selection
        # action = 0
        # for index, val in enumerate(self.R):
        #     action += self.M[index]*val
        # action = int(action)
        # if action >= self.action_set_length:
        #         action = self.action_set_length - 1
        action = self.M[np.argmax(self.R)]
        return action

    # Q(s,a) = Sum of (degree_of_truth_values[i]*q[i, a])
    def calculate_q_value(self):
        q_curr = 0
        for index, truth_value in enumerate(self.R):
            q_curr += truth_value * self.q_table[index, self.M[index]]
        self.Q.append(q_curr)

    # V'(s) = sum of (degree of truth values*max(q[i, a]))
    def calculate_state_value(self):
        v_curr = 0
        for index, rule in enumerate(self.q_table):
            v_curr += (self.R[index] * max(rule))
        self.V.append(v_curr)

    # Q(i, a) += beta*degree_of_truth*delta_Q
    # delta_Q = reward + gamma*V'(s) - Q(s, a)
    def update_q_value(self, reward):
        self.Error = reward + self.gamma * self.V[-1] - self.Q[-1]
        # self.R_ is the degree of truth values for the previous state
        for index, truth_value in enumerate(self.R_):
            delta_q = self.alpha * (self.Error * truth_value)
            self.q_table[index][self.M[index]] += delta_q
        return self

    def save_state_history(self):
        self.R_ = copy.copy(self.R)

    def get_initial_action(self, state):
        self.V.clear()
        self.Q.clear()
        self.truth_value(state)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action

    def get_action(self, state):
        self.truth_value(state)
        action = self.action_selection()
        return action

    def run(self, state, reward):
        self.truth_value(state)
        self.calculate_state_value()
        self.update_q_value(reward)
        action = self.action_selection()
        self.calculate_q_value()
        self.save_state_history()
        return action
