from unityagents import UnityEnvironment
import torch
from collections import deque
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name='Reacher_20_agents/Reacher.exe')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
observation_state_size = brain.vector_observation_space_size
action_space_size = brain.vector_action_space_size
print(observation_state_size)
print(action_space_size)

training_interval = 20
train_steps = 1 #10
agent = Agent(observation_state_size, action_space_size)
scores = []
scores_last_hundred_episodes = deque(maxlen=100)
for episode in range(300):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    states = env_info.vector_observations #states = env_info.vector_observations[0] 
    agent.reset()
    rewards = 0
    for timestep in range(1001):
        actions = agent.select_actions(states)      
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations  #next_state = env_info.vector_observations[0]
        reward = env_info.rewards                  # same
        done = env_info.local_done                 # same
        for agnt in range(20):
            sars = (states[agnt], actions[agnt], reward[agnt], next_state[agnt], done[agnt])
            agent.add(sars)
            if(timestep % training_interval==0):                
                for _ in range(train_steps):
                    agent.train()

        states = next_state
        rewards += np.mean(reward)
        if(np.any(done)):
            break
    scores.append(rewards)
    scores_last_hundred_episodes.append(rewards)
    #if(episode % 100 == 0):
    print('episode {} rewards {:.2f} mean score(100ep) {:.2f}'.format(episode, rewards, np.mean(scores_last_hundred_episodes)))
torch.save(agent.actor.state_dict(), 'actor_checkpoint.pth')
torch.save(agent.actor.state_dict(), 'critic_checkpoint.pth')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()