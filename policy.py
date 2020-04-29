#!/usr/bin/python
#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

# Modified by Immanuel Schwall (manuel.schwall@gmail.com) 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ActorCritic(nn.Module):
	""" Actor Critic neural network with shared body.
	The Actor maps states (actions) to action, log_probs, entropy.
	The Critic maps states to values.
	"""
	
	def __init__(self, state_size, action_size, seed=0):
		""" Initialize the neural net.
        
        Params
        ======
        	state_size: 	dimension of each input state
        	action_size: 	dimension of each output
        	seed: 			random seed
        """
		super().__init__()
		self.seed = torch.manual_seed(seed)
		# fully connected body
		self.fc1_body = nn.Linear(state_size, 64)
		self.fc2_body = nn.Linear(64, 64)
		# actor head
		self.fc3_actor = nn.Linear(64, action_size)
		self.std = nn.Parameter(torch.ones(1, action_size))
		# critic head
		self.fc3_critic = nn.Linear(64, 1)


	def forward(self, state, action=None):
		x = torch.Tensor(state)
		x = F.relu(self.fc1_body(x))
		x = F.relu(self.fc2_body(x))

		# Actor policy
		mean = torch.tanh(self.fc3_actor(x))
		dist = torch.distributions.Normal(mean, F.softplus(self.std))
		
		if action == None:
			action = dist.sample()
		log_prob = dist.log_prob(action)
		log_prob = torch.sum(log_prob, dim=1, keepdim=True)
		entropy = dist.entropy()
		entropy = torch.sum(entropy, dim=1, keepdim=True)

		# Critic value
		value = self.fc3_critic(x)
		return action, log_prob, entropy, value
