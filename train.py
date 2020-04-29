#!/usr/bin/python
from unityagents import UnityEnvironment
from collections import deque
from ppo_agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd



# start unity environment
env = UnityEnvironment(file_name="Tennis.app") # you might have to change the path
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

state_size = env_info.vector_observations.shape[1]
action_size = brain.vector_action_space_size
number_of_agents = len(env_info.agents)

print_every = 10


def main():
    print("Start Training...")
    agent = Agent(state_size, action_size, load_pretrained=False)
    scores = run_ppo(env, brain_name, agent)
    print("\nTraining finished.")

    scores = np.array(scores)
    x = np.where(scores >= 0.5)
    print('The first time a score >= 0.5 was reached at episode {}.'.format(x[0][0]))
    print('Max score reached: {:.4f}'.format(np.amax(scores)))

    df = pd.DataFrame({
        'x': np.arange(len(scores)),
        'y': scores, 
        })
    rolling_mean = df.y.rolling(window=50).mean()

    img_path ="imgs/scores_plot.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(df.x, df.y, label='Scores')
    plt.plot(df.x, rolling_mean, label='Moving avg', color='orange')
    plt.ylabel('Scores')
    plt.xlabel('Episodes')
    plt.legend()
    fig.savefig(fname=img_path)
    print('\nPlot saved to {}.'.format(img_path))


def run_ppo(env, brain_name, agent, num_episodes=2000):
    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, num_episodes+1):
        agent.step(env, brain_name)
        max_score = agent.act(env, brain_name)
        scores.append(max_score)
        scores_window.append(max_score)

        print('\r{}/{} Episode. Current score: {:.4f} Avg last 100 score: {:.4f}'.\
            format(i_episode, num_episodes, max_score, np.mean(scores_window)), end="")
        
        if i_episode % print_every == 0:
            print('\r{}/{} Episode. Current score: {:.4f} Avg last 100 score: {:.4f}'.\
                format(i_episode, num_episodes, max_score, np.mean(scores_window)))

        if np.mean(scores_window) > 0.5:
            agent.save()
            print('\rEnvironment solved after {} episodes. Avg last 100 score: {:.4f}'.\
                format(i_episode, np.mean(scores_window)))
            break

    return scores



if __name__ == "__main__":
    main()