#!/usr/bin/env python3
#
#  gym_discrete.py
#  Example script of using OpenAI Gym in simple grid-world environment
#
import gym
from learning_agent import LearningAgent
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

# How long do we play
NUM_EPISODES = 1000000
# How often we print results
PRINT_EVERY_EPS = 10

#environment = gym.make("FrozenLake-v0")
environment = FrozenLakeEnv(is_slippery=False)

num_states = environment.observation_space.n
num_actions = environment.action_space.n

agent = LearningAgent(num_states, num_actions)

sum_reward = 0

for episode in range(NUM_EPISODES):
    done = False
    last_observation = environment.reset()
    last_action = None
    last_reward = None
    while not done:
        action = agent.get_action(last_observation, environment)
        
        observation, reward, done, info = environment.step(action)
        
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