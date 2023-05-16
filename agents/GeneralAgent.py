# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:50:37 2023

@author: Martin Alejandre Chaya
"""

class GeneralAgent:
    
    def __init__(self):
        self.name = "General Agent"
    
    def choose(self, state, posible_actions):
        return posible_actions[0]
    
    def learn(self, state, action, new_state, reward):
        return
