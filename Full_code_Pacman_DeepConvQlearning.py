import os
import random
import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch.utils.data import DataLoader,TensorDataset
import gymnasium as gym
import ale_py
from PIL import Image
from torchvision import transforms
import cv2 


# Initializing hyperparameters
learning_rate=5e-4
minibatch_size=64
discount_factor=0.99 # gamma of q learning  which if it is close to 1 it will consider future rewards better



class Network(nn.Module):

    def __init__(self,action_size,seed=42): # seed is used for reproducibility and randomness
        super(Network,self).__init__() # inheritance
        self.seed= torch.manual_seed(seed)

        self.conv1= nn.Conv2d(3,32, kernel_size=8 , stride=4)  # 3 is rgb input channels ,32 as output channels is good enough ,kernel size is 8by8
        self.bn1= nn.BatchNorm2d(32) # 32 was the out previously so we are normalizing it 
        self.conv2= nn.Conv2d(32,64, kernel_size=4 , stride=2) 
        self.bn2= nn.BatchNorm2d(64)
        self.conv3= nn.Conv2d(64,64, kernel_size=3 , stride=1) # kernel size is feature extractors where as filter is the number of input and output taking part of the network.
        self.bn3= nn.BatchNorm2d(64)        
        self.conv4= nn.Conv2d(64,128, kernel_size=3 , stride=1) 
        self.bn4= nn.BatchNorm2d(128)


        self.fc1=nn.Linear(10*10*128, 512 ) #the input to this should be based on formula in theory after flattening, the formula is based on image shape
        self.fc2 = nn.Linear(512, 256) # the formula gets o/p size after each layer 
        self.fc3 = nn.Linear(256, action_size)



    def forward(self,state):
        x= F.relu(self.bn1(self.conv1(state))) # just feeding the image of the game pacman and then apply batch normalization and relu
        x= F.relu(self.bn2(self.conv2(x)))
        x= F.relu(self.bn3(self.conv3(x)))
        x= F.relu(self.bn4(self.conv4(x)))

        # x.size(0): Keeps the batch size unchanged. remains the same 
        # -1: Tells PyTorch to automatically get the correct number of features for the second dimension, flattening the rest of the tensor.
        # Ann takes 1 dimensional whereas in CNN we can get 3 (height, width, channels)
        # so when we feed to ANN  we flatten to 1 dimension 
        # To convert a multi-dimensional tensor into a flat vector, we use .view().

        # example

        # Flatten spatial dimensions (2, 3, 4) into one dimension
            # x_reshaped = x.view(5, -1) the -1 reshapes multidimension to 1 dimesnion and the 5 here is batch size
            # x.size(0) below gets the batch size and it is assigned as the batch size
            # print(x_reshaped.shape)  # (5, 24)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


env= gym.make('MsPacmanDeterministic-v0',full_action_space= False)
state_shape=env.observation_space.shape
state_size = env.observation_space.shape[0]
number_actions = env.action_space.n
print('State shape: ', state_shape)
print('State size: ', state_size)
print('Number of actions: ', number_actions)

# Preprocessing the Frames torchVision is used here

def preprocess_frame(frame): # the frame currently is inform of numpy array
  
  frame = Image.fromarray(frame) # now its an image

  # compose class takes only one argument which is a List  # preprocess here is an instance of Compose class
  preprocess = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
  return preprocess(frame).unsqueeze(0) # increase dimension of the batch


class Agent():

  def __init__(self, action_size):
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.action_size = action_size
    self.local_qnetwork = Network(action_size).to(self.device)
    self.target_qnetwork = Network(action_size).to(self.device)
    self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr = learning_rate)
    self.memory = deque(maxlen = 10000)

  def step(self, state, action, reward, next_state, done):
    state = preprocess_frame(state) # since its an image pytorch tensor
    next_state = preprocess_frame(next_state)  # since its an image pytorch tensor
    self.memory.append((state, action, reward, next_state, done))
    if len(self.memory) > minibatch_size:
      experiences = random.sample(self.memory, k = minibatch_size) # random sample from the experience replay
      self.learn(experiences, discount_factor)

  def act(self, state, epsilon = 0.):
    state = preprocess_frame(state).to(self.device)
    self.local_qnetwork.eval()
    with torch.no_grad():
      action_values = self.local_qnetwork(state)
    self.local_qnetwork.train()
    if random.random() > epsilon:
      return np.argmax(action_values.cpu().data.numpy())
    else:
      return random.choice(np.arange(self.action_size))


# previously we do not have replay memory class we have to implement stack of states etc here
  def learn(self, experiences, discount_factor): 

    # we need to get the separate values into the variable through iterating experiences the * is for iterating 
    # in simple words we are unzipping from experiences and storing the values in vairables, zip * means unzipping
    # the state and next state are already Pytorch tensors due to preprocess function call in step function
    # the np.vstack can take tensors also , not only numpy array
    # the np.vstack is again sent torch.fromnumpy so that we can again convert it to tensors

    states, actions, rewards, next_states, dones = zip(*experiences) 

    states = torch.from_numpy(np.vstack(states)).float().to(self.device) # since image here we do not iterate through the experiences like before
    actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
    rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
    next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
    dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

    next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
    q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
    q_expected = self.local_qnetwork(states).gather(1, actions)
    loss = F.mse_loss(q_expected, q_targets)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

# Initializing the DCQN agent

agent = Agent(number_actions)

# Training the DCQN agent

number_episodes= 2000
maximum_number_timestamps_per_episode=10000
epsilon_starting_value=1.0
epsilon_ending_value=0.01
epsilon_decay_value=0.995
epsilon= epsilon_starting_value
scores_on_100_episodes=deque(maxlen=100)


for episode in range(1,number_episodes+1):
  state,_= env.reset()
  score=0
  print("running")
  for t in range(maximum_number_timestamps_per_episode):
    action=agent.act(state,epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  scores_on_100_episodes.append(score)
  epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_on_100_episodes)))
  if np.mean(scores_on_100_episodes) >= 500.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(scores_on_100_episodes)))
    torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
    break



# Function to visualize the model's performance and record a video
def show_video_of_model(agent, env_name='MsPacmanDeterministic-v0', output_filename='output_video.mp4'):
    env = gym.make(env_name, render_mode='rgb_array')  # Create environment
    state, _ = env.reset()
    done = False
    
    # Get frame dimensions
    frame = env.render()
    height, width, layers = frame.shape
    
    # Initialize video writer
    video_writer = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (width, height)
    )
    
    try:
        while not done:
            frame = env.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
            video_writer.write(frame_bgr)  # Write frame to video
            
            action = agent.act(state)  # Get action from agent
            state, reward, done, _, _ = env.step(action)  # Apply action in environment
    
    finally:
        env.close()  # Ensure environment is properly closed
        video_writer.release()  # Release video writer
    
    print(f"Video saved as {output_filename}")

show_video_of_model(agent, 'MsPacmanDeterministic-v0')


# for jupyternotebook

# import glob
# import io
# import base64
# import imageio
# from IPython.display import HTML, display
# from gym.wrappers.monitoring.video_recorder import VideoRecorder

# def show_video_of_model(agent, env_name):
#     env = gym.make(env_name, render_mode='rgb_array')
#     state, _ = env.reset()
#     done = False
#     frames = []
#     while not done:
#         frame = env.render()
#         frames.append(frame)
#         action = agent.act(state)
#         state, reward, done, _, _ = env.step(action)
#     env.close()
#     imageio.mimsave('video.mp4', frames, fps=30)

# show_video_of_model(agent, 'MsPacmanDeterministic-v0')

# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")

# show_video()


