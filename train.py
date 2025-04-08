import argparse
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import random
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from MuZero.Env_Runner import Env_Runner
from MuZero.Env_Wrapper import Env_Wrapper
from MuZero.Agent import MuZero_Agent
from MuZero.Networks import Representation_Model, Dynamics_Model, Prediction_Model
from MuZero.Experience_Replay import Experience_Replay

def train():
    # Define device in one place - use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    
    # Setup TensorBoard writer
    log_dir = os.path.join("logs", "muzero_" + time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir)
    
    # Lists to store metrics for plotting
    episode_rewards = []
    losses = []
    
    history_length = 3
    num_hidden = 50
    num_simulations = 20
    replay_capacity = 100
    batch_size = 32
    k = 5
    n = 10
    lr = 1e-3
    value_coef = 1
    reward_coef = 1
    
    raw_env = gym.make('CartPole-v0')
    num_obs_space = raw_env.observation_space.shape[0]
    num_actions = raw_env.action_space.n
    num_in = history_length * num_obs_space
    
    env = Env_Wrapper(raw_env, history_length)
    
    representation_model = Representation_Model(num_in, num_hidden).to(device)
    dynamics_model = Dynamics_Model(num_hidden, num_actions).to(device)
    prediction_model = Prediction_Model(num_hidden, num_actions).to(device)
    
    agent = MuZero_Agent(num_simulations, num_actions, representation_model, dynamics_model, prediction_model, device=device, dtype=dtype).to(device)

    runner = Env_Runner(env, device=device, dtype=dtype)
    replay = Experience_Replay(replay_capacity, num_actions, device=device, dtype=dtype)
    
    mse_loss = nn.MSELoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    logsoftmax = nn.LogSoftmax()
    
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # Use tqdm for progress bar
    for episode in tqdm(range(2000), desc="Training episodes"):
        
        # act and get data
        trajectory = runner.run(agent)
        # save new data
        replay.insert([trajectory])
        
        # The runner already logs the episode returns to the logger
        # We'll track the episode number for plotting later
        
        #############
        # do update #
        #############
             
        if len(replay) < 15:
            continue
            
        if episode < 250:
           agent.temperature = 1
        elif episode < 300:
            agent.temperature = 0.75
        elif episode < 400:
            agent.temperature = 0.65
        elif episode < 500:
            agent.temperature = 0.55
        elif episode < 600:
            agent.temperature = 0.3
        else:
            agent.temperature = 0.25
        
            
        for i in range(16):
            optimizer.zero_grad()
            
            # get data
            data = replay.get(batch_size,k,n)
            
            # network unroll data
            representation_in = torch.stack([torch.flatten(data[i]["obs"]) for i in range(batch_size)]).to(device).to(dtype) # flatten when insert into mem
            actions = np.stack([np.array(data[i]["actions"], dtype=np.int64) for i in range(batch_size)])
            
            # targets
            rewards_target = torch.stack([torch.tensor(data[i]["rewards"]) for i in range(batch_size)]).to(device).to(dtype)
            policy_target = torch.stack([torch.stack(data[i]["pi"]) for i in range(batch_size)]).to(device).to(dtype)
            value_target = torch.stack([torch.tensor(data[i]["return"]) for i in range(batch_size)]).to(device).to(dtype)
            
            # loss
            loss = torch.tensor(0).to(device).to(dtype)
            
            # agent inital step
            state, p, v = agent.inital_step(representation_in)
            
            #policy mse
            policy_loss = mse_loss(p, policy_target[:,0].detach())
            
            #policy cross entropy
            #policy_loss = torch.mean(torch.sum(- policy_target[:,0].detach() * logsoftmax(p), 1))
            
            value_loss = mse_loss(v, value_target[:,0].detach())
            
            loss += ( policy_loss + value_coef * value_loss) / 2

            # steps
            for step in range(1, k+1):
            
                step_action = actions[:,step - 1]
                state, p, v, rewards = agent.rollout_step(state, step_action)
                
                #policy mse
                policy_loss = mse_loss(p, policy_target[:,step].detach())
                
                #policy cross entropy
                #policy_loss = torch.mean(torch.sum(- policy_target[:,step].detach() * logsoftmax(p), 1))
                
                value_loss = mse_loss(v, value_target[:,step].detach())
                reward_loss = mse_loss(rewards, rewards_target[:,step-1].detach())
                
                loss += ( policy_loss + value_coef * value_loss + reward_coef * reward_loss) / k
         
            loss.backward()
            optimizer.step()
            
            # Log loss to TensorBoard
            writer.add_scalar("Loss/total", loss.item(), episode * 16 + i)
            writer.add_scalar("Loss/policy", policy_loss.item(), episode * 16 + i)
            writer.add_scalar("Loss/value", value_loss.item(), episode * 16 + i)
            writer.add_scalar("Loss/reward", reward_loss.item(), episode * 16 + i)
            
            # Store loss for plotting
            losses.append(loss.item())

    # After training, plot the losses and rewards
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Load and plot rewards from the logger file
    try:
        rewards_data = np.loadtxt("episode_returns.csv", delimiter=',', skiprows=1)
        if rewards_data.size > 0:
            if len(rewards_data.shape) == 1:  # Only one row
                episode_nums = [rewards_data[0]]
                rewards = [rewards_data[1]]
            else:
                episode_nums = rewards_data[:, 0]
                rewards = rewards_data[:, 1]
                
            plt.subplot(2, 1, 2)
            plt.plot(episode_nums, rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
    except Exception as e:
        print(f"Could not load rewards data: {e}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_plots.png'))
    plt.show()
    
    # Close the TensorBoard writer
    writer.close()
    
    print(f"Training completed! Logs saved to {log_dir}")

if __name__ == "__main__":

    train()