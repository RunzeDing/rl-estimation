import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
from scipy.io import loadmat
from numpy import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal


# Replay Buffer

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# Env
class EstiEnv():
    """
    estimation environment
    """
    def __init__(self,data_name):
        
        #================= DATA ========================

        # load measured data
        self.data = loadmat(data_name)
        # external force Fy measurement
        self.measurement = self.data['F_y']
        # distance ground truth
        self.groundtruth = self.data['y_NW']
        # acc
        self.acceleration = self.data['a_y_measurement_NW']
        # counter to get reward
        self.counter = 0
        # length limit
        self.data_length_limit = self.acceleration.size
        # random initial K gain
        self.action = np.array([random.random(),random.random()])
        self.done = False
        #================= SYS MODEL =====================
        
        self.A = torch.tensor([[1.0,0.01],[0.0,1.0]])
        self.B = torch.tensor([[0.0001],[0.01]])
        

    def model_predict(self,x_last,uinput):
        
        x_predict = self.A.mm(x_last) + self.B.mm(uinput)
        return x_predict
    
    def reset(self):

        self.counter = 0
        self.done = False
        p3 = 55.72
        p2 = -30.65
        p1 = 5.66
        p0 = -0.2569

        # initial state t
        x_last = torch.tensor([[self.groundtruth[0,self.counter]],[0]])
        # print('x_last',x_last)
        # model predict t+1
        uinput = torch.tensor([[self.acceleration[0,self.counter]]])
        # print('uinput',uinput)
        x_pre  = self.model_predict(x_last,uinput)
        
        # print('x_pre',x_pre)
        # initial K-gain
        self.action = torch.tensor([[random.random()],[random.random()]])
        # print('action',self.action)
        # measurement model
        H = (p3*x_pre[0]**3 + p2*x_pre[0]**2 + p1*x_pre[0] + p0)
        Y = self.measurement[0,self.counter]

        x_post = x_pre + self.action*(H-Y).item()
        x_post = torch.clamp(x_post, min=-10000, max=10000)
        # print('x_post',x_post)
        state = (self.groundtruth[0,self.counter+1] - x_post[0])
        
        # print('state',state)
        return state,x_post

    def step(self,actioninput,xlast,i):
        self.action = actioninput
        self.counter = i
        
        x_last = xlast
        # print('x_last',x_last)
        p3 = 55.72
        p2 = -30.65
        p1 = 5.66
        p0 = -0.2569

        # model predict t+1
        
        uinput = torch.tensor([[self.acceleration[0,self.counter]]])
        # print('uinput',uinput)
        x_pre  = self.model_predict(x_last,uinput)
        # print('x_pre',x_pre)
        # measurement model
        H = (p3*x_pre[0]**3 + p2*x_pre[0]**2 + p1*x_pre[0] + p0)
        Y = self.measurement[0,self.counter]
        # print((H-Y).item())
        x_post = x_pre + self.action*(H-Y).item()
        x_post = torch.clamp(x_post, min=-10000, max=10000)
        # print('x_post',x_post)
        state = (self.groundtruth[0,self.counter+1] - x_post[0])
        
       
        reward = ((self.groundtruth[0,self.counter+1] - x_post[0]))**2
        # print('reward',reward)

        if i > self.data_length_limit - 4:
            self.done = True
        return state,x_post,reward,self.done

    def echo(self):
        print('measurement:','\n')
        print(self.measurement)
        print('acc:','\n')
        print(self.acceleration)
        print('groudtruth:','\n')
        print(self.groundtruth)
        print(self.action)
        print(self.data_length_limit)


class CriticNetwork(nn.Module):
    def __init__(self, beta =0.0003, input_dims = 1, n_actions = 2, fc1_dims=256, fc2_dims=256,
            name='critic'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        
        self.fc1 = nn.Linear(self.input_dims+n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        

    def forward(self, state, action):
        # print(torch.cat([state, action],0))
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

class ActorNetwork(nn.Module):
    def __init__(self, alpha = 0.0003, input_dims =1, max_action = 2, fc1_dims=256, 
            fc2_dims=256, n_actions=2, name='actor'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        

    def forward(self, state):
        # full connect
        prob = self.fc1(state)
        # activate
        prob = F.relu(prob)
        # full connect
        prob = self.fc2(prob)
        # activate
        prob = F.relu(prob)

        # prob mu
        mu = self.mu(prob)
        # prob sigma
        sigma = self.sigma(prob)
        # trick constraint the varience
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, state):
        mu, sigma = self.forward(state)
        # action prob
        probabilities = Normal(mu, sigma)
        # sample action
        actions = probabilities.sample()
        # normalize action
        action = torch.tanh(actions)*torch.tensor(self.max_action)
        # get action prob
        log_probs = probabilities.log_prob(actions)
        #print('prob1',log_probs)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        #print('prob2',log_probs)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        #print('prob3',log_probs)
        return action, log_probs

# Agent
class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[1],
            env=None, gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=1,H=20):
        self.gamma = gamma
        self.tau = tau
        self.alpha = 1
        self.alpha_rate = 0.0003
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        # actor nn
        self.actor = ActorNetwork()
        # two critic nn trick
        self.critic_1 = CriticNetwork(name='critic_1')
        self.critic_2 = CriticNetwork(name='critic_2')
        # stationary critic trick
        self.critic_1_target = CriticNetwork(name='critic_1_target')
        self.critic_2_target = CriticNetwork(name='critic_2_target')
        self.scale = reward_scale
        self.update_network_parameters()
        self.H = H
        

    def choose_action(self, state):
        actions, _ = self.actor.sample_normal(state)

        return actions

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    def update_network_parameters(self):
        tau = self.tau

        # theta_target = tau*theta + (1-tau)*theta_target
        # 
        # Update critic target 1 
        target_value_params_1 = self.critic_1_target.named_parameters()
        value_params_1 = self.critic_1.named_parameters()

        target_value_state_dict = dict(target_value_params_1)
        value_state_dict = dict(value_params_1)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.critic_1_target.load_state_dict(value_state_dict)

        # theta_target = tau*theta + (1-tau)*theta_target
        # 
        # Update critic target 2 
        target_value_params_2 = self.critic_2_target.named_parameters()
        value_params_2 = self.critic_2.named_parameters()

        target_value_state_dict = dict(target_value_params_2)
        value_state_dict = dict(value_params_2)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.critic_2_target.load_state_dict(value_state_dict)

    def sync_network_parameters(self):

    
        tau = 1
        # Sync params 
        # theta_target = theta
        # 
        # Update critic target 1 
        target_value_params_1 = self.critic_1_target.named_parameters()
        value_params_1 = self.critic_1.named_parameters()

        target_value_state_dict = dict(target_value_params_1)
        value_state_dict = dict(value_params_1)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.critic_1_target.load_state_dict(value_state_dict)

        # theta_target = tau*theta + (1-tau)*theta_target
        # 
        # Update critic target 2 
        target_value_params_2 = self.critic_2_target.named_parameters()
        value_params_2 = self.critic_2.named_parameters()

        target_value_state_dict = dict(target_value_params_2)
        value_state_dict = dict(value_params_2)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.critic_2_target.load_state_dict(value_state_dict)


    def learn(self):
        
        if self.memory.mem_cntr < self.batch_size:
            return

        # sample mini-batch B from replay buffer D
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        # values
        reward = torch.tensor(reward, dtype=torch.float)
        done = torch.tensor(done)
        state_ = torch.tensor(new_state, dtype=torch.float)
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        alpha = torch.tensor(self.alpha,dtype=torch.float,requires_grad = True)

        # sample a' based on current policy and s'
        new_actions, log_probs = self.actor.sample_normal(state_)

        # calculate Q(s',a') using target nn
        q_hat_1 = self.critic_1_target.forward(state_,new_actions)
        q_hat_2 = self.critic_2_target.forward(state_,new_actions)

        # Q(s',a') = min(Q1,Q2)
        q_hat = torch.min(q_hat_1,q_hat_2)
        # TD target = r + gamma(Q(s',a') - log_probs(a'))
        q_target = self.scale*reward.reshape(self.batch_size,1) + self.gamma*(q_hat - self.alpha * log_probs)
        
        # calculate Q(s,a)
        q_value_1 = self.critic_1.forward(state,action)
        q_value_2 = self.critic_2.forward(state,action)

        # calculate residual 
        # Update critic nn
        # Optim Index J = E[0.5*(Q(s,a) - TD target)^2]
        q_loss_1 = 0.5*F.mse_loss(q_value_1,q_target)
        q_loss_2 = 0.5*F.mse_loss(q_value_2,q_target)
        q_loss = q_loss_1 + q_loss_2 
        # update Q(s,a) by q_loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_loss.backward(retain_graph=True)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        # calculate actor loss based on new critic
        # sample actions a, log_probs(a)
        actions, log_probs = self.actor.sample_normal(state)

        # calculate Q(s,a) using new critic nn
        q_1 = self.critic_1.forward(state,actions)
        q_2 = self.critic_2.forward(state,actions)

        # Q(s,a) = min(Q1,Q2)
        q = torch.min(q_1,q_2)

        # Update actor nn
        # Optim Index J_\pi = E[ \alpha * log_probs(a) - Q(s,a) ]
        # minimize J not maximize
        actor_loss = q - self.alpha*log_probs
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        # Updata entropy gain \alpha
        # 
        alpha_loss = torch.mean(-alpha*log_probs - alpha*self.H)
        alpha_loss.backward(retain_graph=True)
        alpha = alpha + self.alpha_rate * alpha.grad
        

# SAC training process
if __name__ == '__main__':

    # build env
    env = EstiEnv('RLDATA.mat')

    # build agent
    agent = Agent()

    # initial sync theta and theta_target
    agent.sync_network_parameters()

    # inital espisode limit for replay buffer
    n_games_initial = 1
    # training episode limit
    n_games = 100
    # score history
    score_history = []

    # inital episode
    for i in range(n_games):
        # initial state
        state,x_post = env.reset()
        done = False
        score = 0
        k = 0
        # policy evaluation (initial replay buffer)
        while not done:
            # evaluate action a
            action = agent.choose_action(state)
            state_,x_post,reward,done = env.step(action.reshape(2,1),x_post,k)
            score += reward
            # append to replay buffer D
            agent.remember(state, action.reshape(1,2), reward, state_, done)
            k+=1
            state = state_
            if i >n_games_initial :
                agent.learn()
            
        score_history.append(score)
        print('episode ', i, 'score %.1f' % score)

    
   