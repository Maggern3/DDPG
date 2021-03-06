from networks import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import random
import copy

class Agent():
    def __init__(self, state_size, action_size):
        super().__init__() 
        gpu = torch.cuda.is_available()
        if(gpu):
            print('GPU/CUDA works! Happy fast training :)')
            torch.cuda.current_device()
            torch.cuda.empty_cache()
            self.device = torch.device("cuda")
        else:
            print('training on cpu...')
            self.device = torch.device("cpu")

        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic = Critic(state_size, action_size).to(self.device)
        self.critic_target = Critic(state_size, action_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0)
        self.replay_buffer = deque(maxlen=1000000)#1m
        self.gamma = 0.95#0.99
        self.batch_size = 128        
        self.tau = 0.001        
        self.seed = random.seed(2)
        self.noise = OUNoise((20, action_size), 2)
        self.target_network_update(self.actor_target, self.actor, 1.0)
        self.target_network_update(self.critic_target, self.critic, 1.0)

    def select_actions(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        self.actor.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def add(self, sars):
        self.replay_buffer.append(sars)

    def train(self):    
        if(len(self.replay_buffer) > self.batch_size): 
            states, actions, rewards, next_states, dones = self.sample()            
            next_actions = self.actor_target(next_states)
            next_state_q_v = self.critic_target(next_states, next_actions)
            #print(next_state_q_v)
            q_targets = rewards + (self.gamma * next_state_q_v * (1-dones))
            current_q_v = self.critic(states, actions)
            critic_loss = F.mse_loss(current_q_v, q_targets)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
            self.critic_optim.step()

            actions = self.actor(states)
            actor_loss = -self.critic(states, actions).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            self.target_network_update(self.actor_target, self.actor, self.tau)
            self.target_network_update(self.critic_target, self.critic, self.tau)

    def target_network_update(self, target_network, network, tau):
        for network_param, target_param in zip(network.parameters(), target_network.parameters()):
            target_param.data.copy_(tau * network_param.data + (1.0-tau) * target_param.data)

    def sample(self):        
        samples = random.sample(self.replay_buffer, k=self.batch_size)     
        states = torch.tensor([s[0] for s in samples]).float().to(self.device)        
        actions = torch.tensor([s[1] for s in samples]).float().to(self.device)
        rewards = torch.tensor([s[2] for s in samples]).float().unsqueeze(1).to(self.device)
        next_states = torch.tensor([s[3] for s in samples]).float().to(self.device)
        dones = torch.tensor([s[4] for s in samples]).float().unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):#0.1,0.08,0.06
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(torch.rand(x.shape))
        self.state = x + dx
        return self.state