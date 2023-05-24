# -*- coding: utf-8 -*-
"""
Created on Sat May 13 11:01:00 2023

@author: Martín Alejandre Chaya
"""

import numpy as np

class IntelligentAgent():
    
    def __init__(self, env, risk):
        self.name = "Intelligent-"+'{:.2f}'.format(risk)
        self.risk = risk
        self.highest_price = env.highest_price
        self.days = env.days
        
    def choose(self, state, posible_actions):
        # No options
        if (len(posible_actions) == 1):
            return posible_actions[0]
        
        # Calculate the probability that will get a better reward than current
        days_left = self.days - state[0] - 1
        # Success is when reward is greater than current reward
        prob_success = 1 - state[1] / self.highest_price
        # Repeeat the experiment days_left times (binomial)
        prob_better = 1 - (1-prob_success)**days_left
        
        # Take the risk of waiting if probability is too low
        if (prob_better > self.risk):
            return 0 # don't sale
        else:
            return 1 # sale
