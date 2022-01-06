import numpy as np 
import gym

import gym.spaces

class ToContinuous(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        return self.env.step([
            np.array([-1.0, 0.0, 0.0]), 
            np.array([1.0, 0.0, 0.0]), 
            np.array([0.0, 0.0, 0.8]), 
            np.array([0.0, 1.0, 0.8]), 
            np.array([0.0, 0.0, 0.0]), 
        ][action])

class MeanRGB(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, obs):
        return np.mean(obs, axis=0, keepdims=True)

class Doom(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.reshape(
            (obs[::5, ::5, 0] * 0.5 + obs[::5, ::5, 2] * 0.5).astype(np.uint8), 
            [48, 64, 1]
        )

class SkipFrames(gym.Wrapper):
    def __init__(self, env, number):
        super().__init__(env)
        self.number = number
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        reward = 0

        for _ in range(self.number):
            obs, rew, done, info = self.env.step(action)
            reward += rew
            if done:break

        return obs, reward, done, info

class Buffer(gym.Wrapper):
    def __init__(self, env, number):

        super().__init__(env)
        self.number = number 
    
    def reset(self):
        obs = self.env.reset()

        self.buffer = [np.zeros_like(obs) for _ in range(self.number)]

        self.buffer[0] = obs 

        return np.concatenate(self.buffer, axis=0)
    
    def step(self, action):
        self.buffer[1:] = self.buffer[:-1]
        obs, reward, done, info = self.env.step(action)
        self.buffer[0] = obs

        return np.concatenate(self.buffer, axis=0), reward, done, info


class DoneIfReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.done = True
        self.obs = None
    
    def reset(self):
        if self.done:
            self.obs = self.env.reset()
        return self.obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.done = done 

        return obs, reward, done or reward != 0, info
        

class Frameskip(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        reward = 0 

        for _ in range(4):
            obs, re, done, info = self.env.step(action)
            reward += re 
            if done:break 
        
        return obs, reward, done, info

class FromUInt8(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

class Transpose(gym.ObservationWrapper):
    def __init__(self, env, axis=[2, 0, 1]):
        super().__init__(env)
        self.axis=axis
    
    def observation(self, observation):
        return np.transpose(observation, self.axis)

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.stack = np.zeros([4, 70, 80])
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.stack[1:] = self.stack[:3]

        self.stack[0] = (
            sum([np.mean(obs[i::3, j::2], axis=2) for i in range(3) for j in range(2)]) / 6
        ).astype(np.uint8)

        return self.stack, reward, done, info
    
    def reset(self):
        self.stack = np.zeros([4, 70, 80])
        obs = self.env.reset()

        self.stack[0] = (
            sum([np.mean(obs[i::3, j::2], axis=2) for i in range(3) for j in range(2)]) / 6
        ).astype(np.uint8)

        return self.stack

