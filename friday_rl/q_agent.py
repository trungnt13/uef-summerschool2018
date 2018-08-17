#!/usr/bin/env python3
#
#  agent_sarsa.py
#  Implementation of tabular Q-learning
#
import numpy as np
import math as m
import random as r

DISCOUNT_FACTOR = 0.98
LEARNING_RATE = 0.1
INIT_VALUE = 1.0

class QAgent:
    """ 
    Q-learning agent
    """
    def __init__(self, 
                 num_states, 
                 num_actions, 
                 learning_rate=LEARNING_RATE,
                 discount_factor=DISCOUNT_FACTOR):
        # Table storing Q-values
        self.q = np.ones((num_states, num_actions))*INIT_VALUE
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    def get_action(self, state, env):
        """
        Returns action with highest value
        Parameters:
            state - Int indicating current state
            env - Gym environment
        Returns:
            action - Int indicating optimal action
        """
        return np.argmax(self.q[state])

    def update(self, s, a, r, t, s_prime, a_prime):
        """
        Update Q-values with Q-learning 
        Parameters:
            s,a,r - State, action and reward at time t
            t - If s_prime is terminal
            s_prime, a_prime - State and action at time t+1
        """
        # Q-learning update:
        # Q(s,a) = Q(s,a) + \alpha * [r + \gamma * \max_a Q(s',a) - Q(s,a)]
        target = 0

        # If s_prime is terminal, target is just r
        if t:
            target = r
        else:
            target = (r + 
                      self.discount_factor * np.max(self.q[s_prime])
                      )

        # Update Q-values according to target
        self.q[s,a] += self.learning_rate * (target - self.q[s,a])