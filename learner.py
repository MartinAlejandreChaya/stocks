# -*- coding: utf-8 -*-
"""
Created on Sun May 21 20:24:54 2023

@author: marti
"""
import numpy as np
from numpy import random as rd





def learn_simple(agent, env, episodes, on_policy = True, exploring_starts = False, learning_rate = 0.01, discount_factor = 1., epsilon = 0.1):
    episode_rewards = np.zeros(episodes)
    agent.set_learning_parameters(learning_rate, discount_factor)

    for episode in range(episodes):
        current_state = env.init_state
        accum_reward = 0
        
        agent.begin_learning_episode()
        
        while(current_state != "final"):
            posible_actions = env.get_actions(current_state)
            forced_action = len(posible_actions) == 1
            exploring_action = rd.rand() < epsilon
            if (exploring_action):
                chosen_action = rd.choice(posible_actions)
            else:
                chosen_action = agent.choose(current_state, posible_actions)
            new_state, reward = env.act(current_state, chosen_action)
            
            agent.learn(current_state, chosen_action, reward, new_state, exploring_action, forced_action)
            
            current_state = new_state
            accum_reward += reward
            
        episode_rewards[episode] = accum_reward
        
    return episode_rewards
    
    
