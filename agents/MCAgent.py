# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:02:39 2023

@author: marti
"""


import numpy as np

from agents.LearningAgent import LearningAgent

class MCAgent(LearningAgent):
    
    def __init__(self, env, name = ""):
        super(MCAgent, self).__init__("MonteCarlo", env, name_extra = name)
    
    
    # Starts learning episode
    def begin_learning_episode(self):
        self.state_action_reward_tuples = []
    # Setup learning parameters
    def set_learning_parameters(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    # Update state-action values based on one interation
    def learn(self, state, action, reward, next_state, exploring_action, forced_action):
        self.state_action_reward_tuples.append((state, action, reward, exploring_action, forced_action))
            
        if (next_state == "final"):
            accum_reward = 0
            # Actually lear
            for i in range(len(self.state_action_reward_tuples)):
                # Retrieve tuple
                (st, act, rew, explor, forced) = self.state_action_reward_tuples[-1-i]
                # Get step actual return
                accum_reward = self.discount_factor*accum_reward + rew
                
                if (forced): # Skip forced actions
                    continue
                
                # Update state-action value
                self.update_state_action_value(st, act, accum_reward, explor)
    
        