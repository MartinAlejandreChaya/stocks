# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:54:52 2023

@author: Martín Alejandre Chaya
"""


import numpy as np

"""
Sells whenever price surpases initial stock price
"""
class ImpatientAgent():
    
    def __init__(self, env):
        self.name = "Impatient"
        self.init_price = env.init_price
        
    def choose(self, state, posible_actions):
        # No options
        if (len(posible_actions) == 1):
            return posible_actions[0]
        
        # Whenever price is higher than initial
        if (state[1] >= self.init_price):
            return 1
        else:
            return 0
        
