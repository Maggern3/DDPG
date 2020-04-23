from unityagents import UnityEnvironment
import torch
from collections import deque
from agent import Agent
import numpy as np

env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size
print(observation_state_size)
print(action_space_size)

training_interval = 4

agent = Agent(observation_state_size, action_space_size)
scores = deque(maxlen=100)
for episode in range(150):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations[0] 
    agent.reset()
    rewards = 0
    for timestep in range(300000):
        actions = agent.select_actions(state)    
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward    
        done = env_info.local_done[0]  
        if(timestep % training_interval==0):
            agent.train()
        sars = (state, actions, reward, next_state, done)
        agent.add(sars)
        state = next_state
        rewards += reward
        if(done):
            break
    scores.append(rewards)
    
    #if(episode % 100 == 0):
    print('episode {} rewards {:.2f} mean score {:.2f}'.format(episode, rewards, np.mean(scores)))