# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:54:52 2023

@author: Martín Alejandre Chaya
"""

from agents.GeneralAgent import GeneralAgent

import numpy as np

"""
Sells whenever price surpases initial stock price
"""
class ImpatientAgent(GeneralAgent):
    
    def __init__(self):
        self.name = "Impatient"
        
    def choose(self, state, posible_actions):
        if (len(posible_actions) == 1):
            return posible_actions[0]
        elif (state[1] >= 10):
            return 1
        else:
            return 0
        
    
    def learn(self, state, action, new_state, reward):
        return