#!/usr/bin/env python3
#
#  agent_sarsa.py
#  Implementation of tabular Q-learning
#
import math as m
import random as r

class LearningAgent:
    """ 
    A baseclass for learning agent
    """
    def __init__(self, 
                 num_states, 
                 num_actions):
        self.num_actions = num_actions
    
    def get_action(self, state, env):
        """
        Returns action
        Parameters:
            state - Int indicating current state
            env - Gym environment
        Returns:
            action - Int indicating optimal action
        """
        return r.randint(0,self.num_actions-1)

    def update(self, s, a, r, t, s_prime, a_prime):
        """
        Update agent
        Parameters:
            s,a,r - State, action and reward at time t
            t - If s_prime is terminal
            s_prime, a_prime - State and action at time t+1
        """
        return