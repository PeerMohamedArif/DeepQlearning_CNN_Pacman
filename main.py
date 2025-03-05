# main.py - Main training script

import gymnasium as gym
import numpy as np
import torch
from agent import Agent
from environment import preprocess_frame
from config import *
from utils import show_video_of_model
from collections import deque

# Initialize environment
env = gym.make(ENV_NAME, full_action_space=False)
state_shape = env.observation_space.shape
number_actions = env.action_space.n

print('State shape:', state_shape)
print('Number of actions:', number_actions)

# Initialize agent
agent = Agent(number_actions)

# Training loop
scores_on_100_episodes = deque(maxlen=100)
epsilon = EPSILON_START

for episode in range(1, NUMBER_EPISODES + 1):
    state, _ = env.reset()
    score = 0
    for t in range(MAX_TIMESTEPS):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    
    scores_on_100_episodes.append(score)
    epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end="")
    
    if episode % 100 == 0:
        print(f'\nEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
    
    if np.mean(scores_on_100_episodes) >= 500.0:
        print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break

# Show agent performance
show_video_of_model(agent, ENV_NAME)
