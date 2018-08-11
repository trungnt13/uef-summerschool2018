#!/usr/bin/env python3
#
#  gym_discrete.py
#  Example script of using OpenAI Gym in simple grid-world environment
#
import gym
from time import sleep

SLEEP_TIME = 0.05

class Agent:
    """ 
    This is our agent, which decides which actions to take based on given
    environment observation.
    """
    def __init__(self):
        # Nothing interesting here (yet)
        pass
    
    def get_action(self, observation, environment):
        """
        Returns an action for given observation 
        
        Parameters:
            observation: Current observation from environment
            environment: Environment object itself, for e.g. accessing 
                         action_space
        Returns:
            action: An action to take next. See environment.action_space for 
                    type
        """
        # For now we just have random agent: Take random action at each step
        return environment.action_space.sample()

agent = Agent()

# Only change is here 
# Task: Reach goal at the top-right
environment = gym.make("MountainCarContinuous-v0")

observation = environment.reset()

done = False

environment.render()
sleep(SLEEP_TIME)

step_ctr = 0

while not done:    
    action = agent.get_action(observation, environment)
    
    observation, reward, done, info = environment.step(action)
    step_ctr += 1
    
    print("Step: {}\t Action: {:.3f}\t Reward: {:.3f}\t Done: {}".format(
          step_ctr, action[0], reward, done))
    environment.render()
    sleep(SLEEP_TIME)

# Close the environment to clean up
environment.close()