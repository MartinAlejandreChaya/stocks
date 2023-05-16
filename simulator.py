# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:23:52 2023

@author: Martín Alejandre Chaya
"""
from environments.SimpleEnvironment import SimpleEnvironment
from agents.RandomAgent import RandomAgent
from agents.ImpatientAgent import ImpatientAgent
from agents.IntelligentAgent import IntelligentAgent

import numpy as np
import matplotlib.pyplot as plt

def simulate_run(agent, environment, verbose = 2):
    if (verbose >= 2):
        print("Simulating a run of agent", agent.name, "on environment", environment.name)
    
    current_state = environment.init_state
    acum_reward = 0
    
    while (not current_state == "final"):
        posible_actions = environment.get_actions(current_state)
        chosen_action = agent.choose(current_state, posible_actions)
        new_state, reward = environment.act(current_state, chosen_action)
        agent.learn(current_state, chosen_action, new_state, reward)
        
        acum_reward += reward
        
        if (verbose >= 3):
            print("\t state, action, reward:", current_state, chosen_action, reward)
        
        current_state = new_state
        
    if (verbose >= 2):
        print("Accumulated reward: ", acum_reward)
        
    return acum_reward


def simulate_runs(agent, environment, runs, verbose = 1):
    
    if (verbose >= 1):
        print("Simulating", runs, "runs of agent", agent.name, "on environment", environment.name)
    
    run_rewards = np.zeros(runs)
    
    for run in range(runs):
        if (verbose == 2):
            print("Run", (run+1), "/", runs)
        run_reward = simulate_run(agent, environment, verbose)
        run_rewards[run] = run_reward
    
    if (verbose >= 1):
        print(run_rewards)
        
    return run_rewards
  

def smooth_line(rewards, smooth_weight = 0.1):
    smoothed = np.zeros(rewards.shape)
    smoothed[0] = rewards[0]
    for i in range(1, rewards.shape[0]):
        smoothed[i] = smoothed[i-1] + smooth_weight*(rewards[i] - smoothed[i-1])
    return smoothed
    
def plot_agents_rewards(agents, environment, runs, smooth = True, smooth_weight = 0.1):
    avgs = {}
    stds = {}
    for agent in agents:
        run_rewards = simulate_runs(agent, environment, runs, verbose=0)
        if (smooth):
            run_rewards = smooth_line(run_rewards, smooth_weight)
        plt.plot(run_rewards, label = agent.name)
        avgs[agent.name] = np.mean(run_rewards)
        stds[agent.name] = np.std(run_rewards)
    
    plt.legend()
    plt.title(environment.name)
    plt.xlabel("Run")
    plt.ylabel("Reward")
    plt.show()
    
    # Order the agents by average reward
    agents.sort(key = lambda x: avgs[x.name], reverse = True)
    print('\n{:<5} {:<20} {:<6} {:<6}'.format("Rank","Agent","Mean","Deviation"))
    for i, agent in enumerate(agents):
        line = '{:<5} {:<20} {:<6.2f} {:<6.2f}'.format((i+1), agent.name, avgs[agent.name], stds[agent.name])
        print(line)
    
    
    
    
    
simple_environment = SimpleEnvironment()
random_agent = RandomAgent()
impatient_agent = ImpatientAgent()
intelligent_agent = IntelligentAgent(risk = 0.5)


# plot_agents_rewards([random_agent, impatient_agent, intelligent_agent], simple_environment, 1000)


intelligent_agents = [IntelligentAgent(risk = i/10) for i in range(11)]

# plot_agents_rewards(intelligent_agents, simple_environment, 10000, smooth_weight=0.003)

intelligent_agents = [IntelligentAgent(risk = 0.6 + i/50) for i in range(11)]

# plot_agents_rewards(intelligent_agents, simple_environment, 20000, smooth_weight=0.003)

intelligent_agent = IntelligentAgent(risk = 0.74)

plot_agents_rewards([random_agent, impatient_agent, intelligent_agent], simple_environment, 10000, smooth_weight = 0.02)


