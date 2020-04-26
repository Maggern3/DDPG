from unityagents import UnityEnvironment
import torch
from collections import deque
from agent import Agent
import numpy as np

env = UnityEnvironment(file_name='Reacher_20_agents/Reacher.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size
print(observation_state_size)
print(action_space_size)

training_interval = 20
train_steps = 10
agent = Agent(observation_state_size, action_space_size)
scores = deque(maxlen=100)
for episode in range(150):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations #states = env_info.vector_observations[0] 
    agent.reset()
    rewards = 0
    for timestep in range(300000):
        #actions = []
        #for agnt in range(20):
           # actions.append(agent.select_actions(states[agnt]))
        actions = agent.select_actions(states)        
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations  #next_state = env_info.vector_observations[0]
        reward = env_info.rewards                  # same
        done = env_info.local_done                 # same
        if(timestep % training_interval==0):
            for _ in range(train_steps):
                agent.train()
        for agnt in range(20):
            sars = (states[agnt], actions[agnt], reward[agnt], next_state[agnt], done[agnt])
            agent.add(sars)
        states = next_state
        rewards += np.mean(reward)
        if(np.any(done)):
            break
    scores.append(rewards)
    
    #if(episode % 100 == 0):
    print('episode {} rewards {:.2f} mean score {:.2f}'.format(episode, rewards, np.mean(scores)))