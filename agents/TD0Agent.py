# -*- coding: utf-8 -*-
"""
Created on Wed May 24 11:03:32 2023

@author: marti
"""


import numpy as np

from agents.LearningAgent import LearningAgent

class TD0Agent(LearningAgent):

    def __init__(self, env, name = ""):
        super(TD0Agent, self).__init__("TD0", env, name_extra = name)
    
        
    
    # Starts learning episode
    def begin_learning_episode(self):
        self.prev_state = -1
        self.prev_action = -1
        self.prev_reward = -1
    # Setup learning parameters
    def set_learning_parameters(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    # Update state-action values based on one interation
    def learn(self, state, action, reward, next_state, exploring_action, forced_action):
        
        
        if ((not forced_action) and (self.prev_state != -1)):
            actual_return = self.prev_reward + self.discount_factor * self.state_action_space[state[0], state[1], action]
            self.update_state_action_value(self.prev_state, self.prev_action, actual_return)
        
        self.prev_state = state
        self.prev_action = action
        self.prev_reward = reward
        
        if ((not forced_action) and (next_state == "final")):
            self.update_state_action_value(self.prev_state, self.prev_action, self.prev_reward)
        
            