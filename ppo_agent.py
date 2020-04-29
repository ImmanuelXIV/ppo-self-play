#!/usr/bin/python
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Jeremi Kaczmarczyk (jeremi.kaczmarczyk@gmail.com) 2018 
# Modified by Andrei Li (andreiliphd@gmail.com) 2019
# Modified by Immanuel Schwall (manuel.schwall@gmail.com) 2019

from collections import deque
from policy import ActorCritic
import numpy as np
import torch
import random
import torch
import torch.nn as nn
import torch.optim as optim
import random


SEED = 0                    # seed
LR = 5e-4                   # leanring rate for actor critic model
T_MAX_ROLLOUT = 1024        # maximum number of time steps per episode
GAMMA = 0.999               # discount factor for returns
TAU = 0.95					# gae (generalized advantage estimation) param
K_EPOCHS = 16				# optimize surrogate loss with K epochs
BATCH_SIZE = 64				# minibatch size ≤ T_MAX_ROLLOUT
EPSILON_PPO = 0.2           # clipping parameter for PPO surrogate
USE_ENTROPY = False         # apply entropy term y/n
ENTROPY_WEIGHT = 0.01       # coefficient for entropy term
GRADIENT_CLIPPING = 2       # gradient clipping norm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Agent():
    """ Actor Critic agent that implements the PPO algorithm 
    based on Schulman et al. (2017) https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, state_size, action_size, load_pretrained=False):
        """ Initialize the agent.
        Params
        ======
            state_size:     dimension of each state
            action_size:    dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        self.ac_model = ActorCritic(state_size, action_size, seed=SEED)
        self.ac_model_optim = optim.Adam(self.ac_model.parameters(), lr=LR)
        random.seed(SEED)
        print('Number of trainable actor critic model parameters: ', \
        	self.count_parameters())

        if load_pretrained:
        	print('Loading pre-trained actor critic model from checkpoint.')
        	self.ac_model.load_state_dict(torch.load("checkpoints/ac_model.pth", \
        		map_location=torch.device(DEVICE)))


    def count_parameters(self):
        return sum(p.numel() for p in self.ac_model.parameters() if p.requires_grad)


    def act(self, env, brain_name):
        """ Run current action policy and check which score it is reaching.
        Params
        ======
            env:            unity Tennis environment
            brain_name:     environment brain name
        """
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        states1 = states[0]
        states2 = states[1]
        scores = np.zeros(2)
        self.ac_model.eval()

        while True:
            with torch.no_grad():
                # self-play: the same actor critic model is used for two players
                actions1, _, _, _ = self.ac_model(states1)
                actions2, _, _, _ = self.ac_model(states2)
            actions = torch.cat((actions1, actions2), dim=0)
            env_info = env.step([actions.cpu().numpy()])[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            states1 = states[0]
            states2 = states[1]

            if np.any(dones):
                break
        self.ac_model.train()
        return np.max(scores)


    def step(self, env, brain_name):
        """ Collect trajectories/episodes and invoke learning step.
        Params
        ======
            env:            unity Tennis environment
            brain_name:     environment brain name
        """
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        states1 = states[0]
        states2 = states[1]

        trajectory1 = deque()    # trajectory of player 1
        trajectory2 = deque()    # trajectory of player 2

        for k in range(T_MAX_ROLLOUT):
            with torch.no_grad():
                # self-play: the same actor critic model is used for two players
                actions1, log_probs1, _, values1 = self.ac_model(states1)
                actions2, log_probs2, _, values2 = self.ac_model(states2)
            actions = torch.cat((actions1, actions2), dim=0)
            env_info = env.step([actions.cpu().numpy()])[brain_name]
            next_states1 = env_info.vector_observations[0]
            next_states2 = env_info.vector_observations[1]

            rewards = env_info.rewards
            rewards1 = np.array(rewards[0]).reshape([1])
            rewards2 = np.array(rewards[1]).reshape([1])

            dones = np.array(env_info.local_done).astype(np.uint8)

            if np.any(dones):
                dones1 = np.array([1.])
                dones2 = np.array([1.])
            else:
                dones1 = np.array([0.])
                dones2 = np.array([0.])

            trajectory1.append(
                [states1, values1, actions1, log_probs1, rewards1, 1 - dones1])
            trajectory2.append(
                [states2, values2, actions2, log_probs2, rewards2, 1 - dones2])
            
            states1 = next_states1
            states2 = next_states2

        pending_value1 = self.ac_model(states1)[-1]
        pending_value2 = self.ac_model(states2)[-1]
        trajectory1.append([states1, pending_value1, None, None, None, None])
        trajectory2.append([states2, pending_value2, None, None, None, None])

        # self-play: the same actor critic model is used for two players
        self.learn(trajectory1, pending_value1)
        self.learn(trajectory2, pending_value2)

        trajectory1.clear()
        trajectory2.clear()


    def learn(self, trajectory, pending_value):
        """ Make PPO learning step. 
        Params
        ======
            trajectory:     trajectory/episode
            pending_value:  pendig critic value from last state
        """
        storage = deque()
        advantages = torch.Tensor(np.zeros((1, 1)))
        returns = pending_value.detach()

        for i in reversed(range(len(trajectory) - 1)):
            states, value, actions, log_probs, rewards, dones = trajectory[i]
            states = torch.Tensor(states).resize_(1, 24)
            actions = torch.Tensor(actions)
            rewards = torch.Tensor(rewards).unsqueeze(1)
            next_value = trajectory[i + 1][1]
            dones = torch.Tensor(dones).unsqueeze(1)
            returns = rewards + GAMMA * dones * returns

            # calculate generalized advantage estimation
            td_error = rewards + GAMMA * dones * next_value.detach() - value.detach()
            advantages = advantages * TAU * GAMMA * dones + td_error
            storage.append([states, actions, log_probs, returns, advantages])

        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*storage))
        advantages = (advantages - advantages.mean()) / advantages.std()

        storage.clear()
        dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        dataiter = iter(dataloader)

        # update the actor critic modelK_EPOCHS times
        for _ in range(K_EPOCHS):
            # sample states, actions, log_probs_old, returns, advantages
            sampled_states, sampled_actions, sampled_log_probs_old, sampled_returns, sampled_advantages = dataiter.next()

            _, log_probs, entropy, values = self.ac_model(sampled_states, sampled_actions)
            ratio = (log_probs - sampled_log_probs_old).exp()
            surrogate = ratio * sampled_advantages
            surrogate_clipped = torch.clamp(ratio, 1.0 - EPSILON_PPO, 1.0 + EPSILON_PPO) * sampled_advantages

            if USE_ENTROPY:
                loss_policy = - torch.min(surrogate, surrogate_clipped).mean(0) - ENTROPY_WEIGHT * entropy.mean()
            else: 
                loss_policy = - torch.min(surrogate, surrogate_clipped).mean(0)
            
            loss_value = 0.5 * (sampled_returns - values).pow(2).mean()
            loss_total = loss_policy + loss_value
            self.ac_model_optim.zero_grad()
            loss_total.backward()
            nn.utils.clip_grad_norm_(self.ac_model.parameters(), GRADIENT_CLIPPING)
            self.ac_model_optim.step()

            del loss_policy
            del loss_value
            

    def save(self):
        torch.save(self.ac_model.state_dict(), "checkpoints/ac_model.pth")



