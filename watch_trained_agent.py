#!/usr/bin/python
import numpy as np
import torch
from unityagents import UnityEnvironment
from ppo_agent import Agent


# start unity environment
env = UnityEnvironment(file_name="Tennis.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
state_size = env_info.vector_observations.shape[1]
action_size = brain.vector_action_space_size

agent = Agent(state_size, action_size, load_pretrained=True)
num_episodes = 2

for i_episode in range(1, num_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    states1 = states[0]
    states2 = states[1]
    scores = np.zeros(2)
    agent.ac_model.eval()

    while True:
        with torch.no_grad():
            # self-play: same actor critic model is used for two players
            actions1, _, _, _ = agent.ac_model(states1)
            actions2, _, _, _ = agent.ac_model(states2)
        actions = torch.cat((actions1, actions2), dim=0)
        env_info = env.step([actions.cpu().numpy()])[brain_name]
        next_states = env_info.vector_observations
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        states1 = states[0]
        states2 = states[1]

        if np.any(dones):
        	print('Episode {} finished. Scores reached: {}'.format(i_episode, scores))
        	break
env.close()
