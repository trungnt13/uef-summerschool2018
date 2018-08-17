#!/usr/bin/env python3
#
#  training_continuous.py
#  Example script of training agents on continuous tasks
#
import gym
import numpy as np
from time import sleep
from learning_agent import LearningAgent
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# How long do we play
NUM_EPISODES = 1000
# How many episodes we render
SHOW_EPISODES = 5
# How often we print results
PRINT_EVERY_EPS = 10
# How many bins we have per dimension
N_BINS = 10

def cont_to_bin(x, bins):
    """ Returns index of closest bin to x """
    # There's probably easier way for this, but oh well
    return np.argmin(abs(x-bins))

def cont_to_disc(xs, bin_list):
    """ 
    Turns continuous variables in xs to discrete number
    specified by bins. Assumes bins are of equal length
    """
    ret = sum([cont_to_bin(xs[i], bins)*(len(bin_list[0])**i) for i,bins in enumerate(bin_list)])
    return ret

# Observation space is [position, velocity]
# Reward is done*100 - amount of power used
environment = gym.make("MountainCarContinuous-v0")

# Create bins
# Hardcoded for MountainCarContinuous-v0
obs_space = environment.observation_space
act_space = environment.action_space
state_bins = [
        np.linspace(obs_space.low[0],obs_space.high[0], num=N_BINS),
        np.linspace(obs_space.low[1],obs_space.high[1], num=N_BINS)
]
action_bins = np.linspace(act_space.low[0],act_space.high[0], num=N_BINS)

num_states = N_BINS**2
num_actions = N_BINS

#agent = SarsaAgent(num_states, num_actions)
agent = QAgent(num_states, num_actions)

sum_reward = 0

for episode in range(NUM_EPISODES):
    done = False
    last_observation = environment.reset()
    last_observation = cont_to_disc(last_observation, state_bins)
    last_action = None
    last_reward = None
    while not done:
        action = agent.get_action(last_observation, environment)

        observation, reward, done, info = environment.step([action_bins[action]])
        observation = cont_to_disc(observation, state_bins)

        if last_action is not None:
            agent.update(last_observation, action, 
                reward, done, observation, 
                agent.get_action(observation, environment)
            )

        last_observation = observation
        last_action = action
        sum_reward += reward
    
    if ((episode+1) % PRINT_EVERY_EPS) == 0:
        print("Episode %d: %f" % (episode, sum_reward/PRINT_EVERY_EPS))
        sum_reward = 0

# Visualize couple of games
for episode in range(SHOW_EPISODES):
    observation = environment.reset()
    observation = cont_to_disc(observation, state_bins)
    sum_r = 0
    done = False
    while not done:
        action = agent.get_action(observation, environment)

        observation, reward, done, info = environment.step([action_bins[action]])
        observation = cont_to_disc(observation, state_bins)

        sum_r += reward

        environment.render()
        sleep(0.05)
    print("Reward: %f" % sum_r)