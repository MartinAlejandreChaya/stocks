# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:36:58 2023

@author: marti
"""
from agents.GeneralAgent import GeneralAgent

import numpy as np

class RandomAgent(GeneralAgent):
    
    def __init__(self):
        self.name = "Random"
        
    def choose(self, state, posible_actions):
        return np.random.choice(posible_actions)
    
    def learn(self, state, action, new_state, reward):
        return