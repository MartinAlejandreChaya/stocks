# -*- coding: utf-8 -*-
"""
Created on Wed May 24 13:51:30 2023

@author: marti
"""

class LearningAgent():
    
    def __init__(self, name, env, name_extra):
        self.name = name
        if (name_extra != ""):
            self.name += name_extra
        self.state_action_space = env.get_state_action_space()
        self.best_posible_reward = env.highest_price - env.init_price
        
    def choose(self, state, posible_actions):
        # No options
        if (len(posible_actions) == 1):
            return posible_actions[0]
        
        action = self.find_best_action(state, posible_actions)
        return action
        
    # Returns the best posible action
    def find_best_action(self, state, posible_actions):
        highest_val = -10000
        best_action = posible_actions[0]
        for action in posible_actions:
            act_val = self.state_action_space[state[0], state[1], action]
            if (act_val > highest_val):
                best_action = action
                highest_val = act_val
        
        return best_action
    
        
    
    # Update state action value torwards passed parameter
    def update_state_action_value(self, state, action, actual_return, explorative = False):
        update_val = self.learning_rate * (actual_return - self.state_action_space[state[0], state[1], action])
        #imp_samp = 1.
        if (explorative): # Importance sampling. Weight 
            pass # What to do when it is explorative?
        self.state_action_space[state[0], state[1], action] += update_val
    
    # Set all state action values
    def set_state_action_values(self, val):
        self.state_action_space = self.state_action_space * 0 + val
        
        
    # AUXILIARS
    def plot_state_action_values(self, plt):
        
        for i in range(2):
            heatmap = plt.imshow(self.state_action_space[:, :, i], vmin=-10, vmax=self.best_posible_reward)
            plt.colorbar(heatmap)
            if (i == 0):
                plt.title(self.name + " -- Action: don't sell")
            else:
                plt.title(self.name + " -- Action: sell")
            plt.show()