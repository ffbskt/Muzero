import numpy as np
import os
import cv2
import gymnasium as gym

class Env_Wrapper(gym.Wrapper):
    # env wrapper for MuZero Cartpole, LunarLander
    
    def __init__(self, env, history_length):
        super(Env_Wrapper, self).__init__(env)
        
        self.history_length = history_length
        self.num_obs_space = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        
    def reset(self):
    
        self.Return = 0
        self.obs_history = []
        
        # Gymnasium API returns (obs, info)
        obs, _ = self.env.reset()
        self.obs_history.append(obs)
        
        return self.compute_observation()
        
        
    def compute_observation(self):
        
        features = np.zeros((self.history_length, self.num_obs_space))
        
        # features 
        current_feature_len = len(self.obs_history)
        
        # Process each observation individually
        for i, obs in enumerate(self.obs_history):
            # Place each observation at the correct position in the features array
            idx = self.history_length - current_feature_len + i
            # Handle the case where obs might be a tuple (array, info_dict)
            if isinstance(obs, tuple) and len(obs) > 0:
                features[idx] = obs[0]
            else:
                features[idx] = obs
        
        return features.flatten().reshape(1,-1)
    
    
    def step(self, action): 
        # Gymnasium API returns (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # add obs and actions to history
        self.add_history(obs)
        
        obs = self.compute_observation()
        
        self.Return += reward
        if done:
            info["return"] = self.Return
            print("Return:",self.Return)
        
        # Return in the format expected by the rest of the code
        return obs, reward, done, info
        
        
    def add_history(self, obs):
    
        if len(self.obs_history) == self.history_length:
            self.obs_history = self.obs_history[1::]
            
            
        self.obs_history.append(obs)