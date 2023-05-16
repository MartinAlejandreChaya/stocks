# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:06:03 2023

@author: Martín Alejandre Chaya
"""
import numpy as np

class SimpleEnvironment:
    
    def __init__(self, days = 7, init_price = 10, highest_price = 20):
        self.name = "Simple environment"
        # Days of the simulation
        self.days = days
        # Initial price of the simulation and highest price
        self.init_price, self.highest_price = init_price, highest_price
        # The state is a duple representing current day and selling price
        self.init_state = (-1, 0) 
        

    # Returns posible actions for a state 
    def get_actions(self, state):
        # If we are in the initial state there are no posible actions
        if (state[0] == -1):
            return [0]
        # If we are in the last day the agent has to sell, whatever the price
        elif (state[0] == self.days-1):
            return [1]
        # Otherwise, the agent can always choose to sell or not
        else:
            return [0, 1]
    
    # Simulates taking an action from a state
    def act(self, state, action):
        # If the action is to sell return the price - the initial price as reward
        # and terminate
        if (action == 1):
            new_state = "final"
            reward = state[1] - self.init_price
        # If the action is not to sell go onto the next day
        else:
            new_state = (state[0]+1, self.get_random_price())
            reward = 0
        
        return new_state, reward
    
    
    # Returns a random price 
    def get_random_price(self):
        price = np.random.randint(0, self.highest_price)
        return price
        