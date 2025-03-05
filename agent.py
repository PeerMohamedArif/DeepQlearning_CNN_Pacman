import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from network import Network
from environment import preprocess_frame
from config import *

class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.local_qnetwork = Network(action_size).to(self.device)
        self.target_qnetwork = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=LEARNING_RATE)
        self.memory = deque(maxlen=MEMORY_SIZE)

    def step(self, state, action, reward, next_state, done):
        state = preprocess_frame(state)
        next_state = preprocess_frame(next_state)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > MINIBATCH_SIZE:
            experiences = random.sample(self.memory, k=MINIBATCH_SIZE)
            self.learn(experiences, DISCOUNT_FACTOR)

    def act(self, state, epsilon=0.):
        state = preprocess_frame(state).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.tensor(np.vstack(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.vstack(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.vstack(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.vstack(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.vstack(dones).astype(np.uint8), dtype=torch.float).to(self.device)

        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()