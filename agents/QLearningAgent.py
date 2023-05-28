# -*- coding: utf-8 -*-
"""
Created on Sun May 28 18:50:39 2023

@author: marti
"""

import numpy as np

from agents.LearningAgent import LearningAgent

class QLearningAgent(LearningAgent):

    def __init__(self, env, name = ""):
        super(QLearningAgent, self).__init__("TD0(Q-learning)", env, name_extra = name)
        
        
    
    # Starts learning episode
    def begin_learning_episode(self):
        pass
    # Setup learning parameters
    def set_learning_parameters(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    # Update state-action values based on one interation
    def learn(self, state, action, reward, next_state, exploring_action, forced_action):
        
        if (forced_action):
            return # do nothig

        if (next_state == "final" or next_state[0] >= self.state_action_space.shape[0]):
            max_value = 0
        else:
            max_value = np.max(self.state_action_space[next_state[0], next_state[1], :])
        
        actual_return = reward + self.discount_factor * max_value
        self.update_state_action_value(state, action, actual_return)
        