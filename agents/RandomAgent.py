# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:36:58 2023

@author: marti
"""

import numpy as np

class RandomAgent():
    
    def __init__(self, env):
        self.name = "Random"
        self.random_day = np.random.randint(0, env.days)
    
        
    def choose(self, state, posible_actions):
        # No options
        if (len(posible_actions) == 1):
            return posible_actions[0]
        
        
        if (self.random_day == state[0]):
            return 1
        else:
            return 0
