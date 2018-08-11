#!/usr/bin/env python3
#
#  gym_discrete.py
#  Example script of using OpenAI Gym in simple grid-world environment
#
import gym
from time import sleep

SLEEP_TIME = 0.3

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

# Create our agent
agent = Agent()

# Create environment
# FrozenLake-v0 is a 2D, grid-like world (grid world) where player has to 
# reach goal. More info here: https://gym.openai.com/envs/FrozenLake-v0/
# You can list available environments with gym.envs.registry.all()
environment = gym.make("FrozenLake-v0")

# Reset environment to initial state,
# and receive initial observation.
observation = environment.reset()

# "done" or "terminal" tells us if game is over, i.e. we have reached
# the terminal state. 
done = False

# Print out the game state for us apes to enjoy
environment.render()
# Wait a moment to give slow ape-brains time to process the information
sleep(SLEEP_TIME)

# A step counter just to keep track of number of steps taken
step_ctr = 0

# Play one game
while not done:    
    # Get an action from our agent
    action = agent.get_action(observation, environment)
    
    # "step" function receives action to take, executes the action 
    # and advances the game by one step.
    # It returns:
    #   - The observation from following state 
    #   - Reward from taking the action
    #   - Boolean on if the return observation is terminal state
    #   - Additional info depending on the environment
    observation, reward, done, info = environment.step(action)
    step_ctr += 1
    
    # Print out info on action, reward and done/terminal info
    print("\nStep: {}\t Action: {}\t Reward: {}\t Done: {}\n".format(step_ctr, 
          action, reward, done))
    # Also print the state
    environment.render()
    sleep(SLEEP_TIME)

# Close the environment to clean up
environment.close()