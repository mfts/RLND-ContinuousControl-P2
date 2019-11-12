import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, 
                 state_size, 
                 action_size,
                 buffer_size,
                 batch_size, 
                 num_agents, 
                 seed, 
                 gamma,
                 tau,
                 lr_actor,
                 lr_critic,
                 weight_decay,
                 update_every,
                 num_updates):
        '''
        ----------------------------------
        Parameters
        
        state_size:   # of states
        action_size:  # of actions
        buffer_size:  size of the memory buffer
        batch_size:   sample minibatch size
        num_agents:   # of agents
        seed:         seed for random
        gamma:        discount rate for future rewards
        tau:          interpolation factor for soft update of target network
        lr_actor:     learning rate of Actor
        lr_critic:    learning rate of Critic
        weight_decay: L2 weight decay
        update_every: update every 20 time steps
        num_updates:  number of updates to the network
        ----------------------------------
        '''
        
        self.action_size = action_size
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay
        self.update_every = update_every
        self.num_updates = num_updates
        self.t_step = 0
        self.seed = random.seed(seed)
        
        # Actor network agent
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        
        # Critic network agent
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # Noise paramter
        self.noise = OUNoise((num_agents,action_size), seed)
        
        # Experience replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        
    def step(self, state, action, reward, next_state, done):
        '''
        Agents takes next step
        - save most recent environment event to ReplayBuffer for each agent
        - load random sample from memory to agent's policy and value network 10 times for every 20 time steps 
        ''' 
        for s,a,r,ns,d in zip(state,action,reward,next_state,done):
            self.memory.add(s,a,r,ns,d)
            
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)
                
    def act(self, state, add_noise=True):
        '''
        Agent selects action based on current state and selected policy
        '''
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
        
    def reset(self):
        self.noise.reset()
        
    def learn(self, experiences, gamma):
        '''
        Agent updates policy and value parameters based on experiences (state, action, reward, next_state, done)
        
        Q_targets = r + gamma * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        '''
        states, actions, rewards, next_states, dones = experiences
        
        #--------- update critic -----------------------#
        # get current Q
        Q_expected = self.critic_local(states, actions)
        # get next action
        next_actions = self.actor_target(next_states)
        # get Qsa_next
        Q_targets_next = self.critic_target(next_states, next_actions)
        # calculate target with reward and Qsa_next
        Q_targets = rewards + (gamma* Q_targets_next * (1-dones)) 
        
        # calculate loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        #--------- update actor ------------------------#
        # computer actor loss
        pred_actions = self.actor_local(states)
        actor_loss = -self.critic_local(states, pred_actions).mean()
        
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #---------- update target networks -------------#
        # update target network parameters
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        
    def soft_update(self, local_model, target_model, tau):
        '''
        Update target network weights gradually with an interpolation rate of TAU
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    ''' Ornstein-Uhlenbeck process '''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()
        
    def reset(self):
        ''' reset to internal state to initial mu '''
        self.state = copy.copy(self.mu)
        
    def sample(self):
        ''' update internal state and return it as a noise sample '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
class ReplayBuffer:
    ''' Experience replay memory '''
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        ''' add learning experiences to memory '''
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        ''' return random batch of experiences from memory '''
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)