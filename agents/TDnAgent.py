# -*- coding: utf-8 -*-
"""
Created on Wed May 24 12:34:16 2023

@author: marti
"""

import numpy as np

from agents.LearningAgent import LearningAgent

class TDnAgent(LearningAgent):

    def __init__(self, env, n, name = ""):
        super(TDnAgent, self).__init__("TD"+str(n), env, name_extra = name)
        self.n = n
    
    
    # Starts learning episode
    def begin_learning_episode(self):
        self.state_action_reward_tuples = []
        
    # Setup learning parameters
    def set_learning_parameters(self, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
    
    # Update state-action values based on one interation
    def learn(self, state, action, reward, next_state, exploring_action, forced_action):
        
        self.state_action_reward_tuples.append((state, action, reward))
        if (next_state == "final"):
            accum_reward = 0
            for i in range(min(len(self.state_action_reward_tuples), self.n)):
                (st, act, rew) = self.state_action_reward_tuples[-i-1]
                accum_reward = accum_reward*self.discount_factor + rew
                if (i == 0 and forced_action):
                    continue
                self.update_state_action_value(st, act, accum_reward)
        elif (len(self.state_action_reward_tuples) > self.n):
            # Update state-action value of -n
            (st, act, rew) = self.state_action_reward_tuples[-self.n]
            actual_return = 0
            for i in range(self.n):
                actual_return = actual_return*self.discount_factor + self.state_action_reward_tuples[-i-1][2]
            actual_return += self.discount_factor**self.n * self.state_action_space[state[0], state[1], action]
            self.update_state_action_value(st, act, actual_return)
    
        