# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:23:52 2023

@author: Martín Alejandre Chaya
"""
from learner import learn_simple
from environments.SimpleEnvironment import SimpleEnvironment
from agents.RandomAgent import RandomAgent
from agents.ImpatientAgent import ImpatientAgent
from agents.IntelligentAgent import IntelligentAgent
from agents.MCAgent import MCAgent
from agents.TD0Agent import TD0Agent
from agents.TDnAgent import TDnAgent

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
    smoothed[0] = 0
    for i in range(1, rewards.shape[0]):
        smoothed[i] = smoothed[i-1] + smooth_weight*(rewards[i] - smoothed[i-1])
    return smoothed
    
def plot_agents_rewards(agents, environment, runs = 10000, smooth = True, smooth_weight = 0.01):
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
    print('\n{:<5} {:<20} {:<6} {:<10} {:<10}'.format("Rank","Agent","Mean","Deviation", "Confidence 96%"))
    for i, agent in enumerate(agents):
        line = '{:<5} {:<20} {:<6.2f} {:<10.2f} {:<10.4f}'.format((i+1), agent.name, avgs[agent.name], stds[agent.name], 1.96*stds[agent.name]/np.sqrt(runs))
        print(line)

def plot_learning_progress(agents, environment, mode = "simple", episodes = 10000, smooth = True, smooth_weight = 0.01, learning_rate = 0.01, discount_factor = 1., epsilon = 0.1):
    for agent in agents:
        if (mode == "simple"):
            episodes_rewards = learn_simple(agent, environment, episodes, learning_rate = learning_rate, discount_factor = discount_factor, epsilon = epsilon)
        if (smooth):
            episodes_rewards = smooth_line(episodes_rewards, smooth_weight)
        plt.plot(episodes_rewards, label = agent.name)
    
    plt.legend()
    plt.title(environment.name + " learning")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
 

#%%
simple_environment = SimpleEnvironment(days=7)
random_agent = RandomAgent(simple_environment)
plot_agents_rewards([random_agent], simple_environment, runs=10000, smooth = True)
#%%
impatient_agent = ImpatientAgent(simple_environment)
plot_agents_rewards([random_agent, impatient_agent], simple_environment, runs=10000)
#%%
intelligent_agent = IntelligentAgent(simple_environment, risk = 0.5)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent], simple_environment, runs=10000)
#%%
intelligent_agents = [IntelligentAgent(simple_environment, risk=i/10.) for i in range(11)]
plot_agents_rewards(intelligent_agents, simple_environment, runs=10000, smooth_weight = 0.001)
#%%
intelligent_agent = IntelligentAgent(simple_environment, risk=0.75)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent], simple_environment, runs=30000, smooth_weight = 0.001)
#%%
mc_agent = MCAgent(simple_environment)
plot_learning_progress([mc_agent], simple_environment, episodes = 1000)
mc_agent.plot_state_action_values(plt)
#%%
plot_learning_progress([mc_agent], simple_environment, episodes = 100000)
mc_agent.plot_state_action_values(plt)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent, mc_agent], simple_environment, runs=30000, smooth_weight = 0.001)
#%%
mc_agent = MCAgent(simple_environment)
mc_agent_2 = MCAgent(simple_environment, name="exp-st")
mc_agent_2.set_state_action_values(10) # Exploring starts
plot_learning_progress([mc_agent, mc_agent_2], simple_environment, episodes = 100000, smooth_weight = 0.001)
mc_agent_2.plot_state_action_values(plt)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent, mc_agent, mc_agent_2], simple_environment, runs=30000, smooth_weight = 0.001)
#%%
td0_agent = TD0Agent(simple_environment)
mc_agent_2 = MCAgent(simple_environment, name="exp-st")
mc_agent_2.set_state_action_values(10) # Exploring starts
plot_learning_progress([mc_agent_2, td0_agent], simple_environment, episodes = 100000, smooth_weight = 0.001, epsilon=0.1, learning_rate=0.01)
td0_agent.plot_state_action_values(plt)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent, mc_agent_2, td0_agent], simple_environment, runs=30000, smooth_weight = 0.001)
#%%
# With td0 exploring starts needs more iteratiosn to converge because of bootstraping
td0_agent_2 = TD0Agent(simple_environment)
mc_agent_2 = MCAgent(simple_environment, name="exp-st")
mc_agent_2.set_state_action_values(10) # Exploring starts
td0_agent_2.set_state_action_values(10)
plot_learning_progress([mc_agent_2, td0_agent_2], simple_environment, episodes = 100000, smooth_weight = 0.001, epsilon=0.1, learning_rate=0.01)
td0_agent.plot_state_action_values(plt)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent, mc_agent_2, td0_agent_2], simple_environment, runs=30000, smooth_weight = 0.001)
#%%
td0_agent = TD0Agent(simple_environment)
tdn_agent = TDnAgent(simple_environment, n=3)
mc_agent_2.set_state_action_values(10) # Exploring starts
plot_learning_progress([mc_agent_2, td0_agent, tdn_agent], simple_environment, episodes = 200000, epsilon = 0.1, smooth_weight = 0.001, learning_rate = 0.01)
tdn_agent.plot_state_action_values(plt)
plot_agents_rewards([random_agent, impatient_agent, intelligent_agent, mc_agent_2, td0_agent, tdn_agent], simple_environment, runs = 30000, smooth_weight = 0.001)
#%%
# Other valeus for n
tdn_agents = [TDnAgent(simple_environment, n=i) for i in range(1, 7)]
plot_learning_progress(tdn_agents, simple_environment, episodes=100000, smooth_weight = 0.001)
plot_agents_rewards(tdn_agents, simple_environment, runs=30000, smooth_weight=0.001)
#%%
simple_environment = SimpleEnvironment(days=7)
td3_agent = TDnAgent(simple_environment, n=3)
plot_learning_progress([td3_agent], simple_environment, episodes=300000, smooth_weight=0.001)
td3_agent.plot_state_action_values(plt)
plot_agents_rewards([td3_agent, mc_agent_2, intelligent_agent], simple_environment, runs = 30000, smooth_weight=0.001)
#%%
simple_environment = SimpleEnvironment(days=21)
intelligent_agent = IntelligentAgent(simple_environment, risk=0.7)
td3_agent = TDnAgent(simple_environment, n=3)
plot_learning_progress([td3_agent], simple_environment, episodes=200000, smooth_weight=0.001)
td3_agent.plot_state_action_values(plt)
plot_agents_rewards([td3_agent, intelligent_agent], simple_environment, runs = 30000, smooth_weight=0.001)
#%%

