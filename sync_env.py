# attention : très lent
from multiprocessing import Process, Pipe

import gym 
import time 

import numpy as np


def run_env(env_fun, inputPipe, outputPipe, m=1, render=False):
    outputPipe, y = outputPipe
    x, inputPipe  = inputPipe
    x.close()
    y.close()

    env_list = []

    for _ in range(m):
        time.sleep(0.3)
        env_list.append(env_fun())
    obs = [env.reset() for env in env_list]

    shape, dtype = obs[0].shape, obs[0].dtype
    outputPipe.send((shape, dtype.name))

    outputPipe.send_bytes(
        np.array(obs, dtype=dtype).tobytes()
    )

    while True:
        actions = inputPipe.recv()
        reward = []
        done = []
        obs = []
        
        for i in range(m):
            if render and i == 0: env_list[i].render()
            o, r, d, _ = env_list[i].step(actions[i])
            if d: o = env_list[i].reset()
            reward.append(r)
            done.append(d)
            obs.append(o)
        
        reward = np.array(reward, dtype=float)
        done = np.array(done, dtype=bool)
        obs = np.array(obs, dtype=dtype)

        outputPipe.send_bytes(reward.tobytes())
        outputPipe.send_bytes(done.tobytes())
        outputPipe.send_bytes(obs.tobytes())


class VecEnv(object):
    def __init__(self, env_fun, n, m=1, render=False):
        inputPipe  = [Pipe() for _ in range(n)]
        outputPipe = [Pipe() for _ in range(n)]

        for i in range(n):
            process = Process(target=run_env, args=(env_fun, outputPipe[i], inputPipe[i], m, render and i == 0))
            process.start()

            inputPipe[i][0].close()
            outputPipe[i][1].close()
        
        self.inputPipe  = [p[1] for p in inputPipe]
        self.outputPipe = [p[0] for p in outputPipe]

        for i in range(n):
            self.shape, self.dtype = self.inputPipe[i].recv()

        self.init_obs = np.zeros((n*m, *self.shape), dtype=self.dtype)
        for i in range(n): self.init_obs[m*i:m*(i+1)] = np.frombuffer(
                self.inputPipe[i].recv_bytes(), dtype=self.dtype).reshape((m, *self.shape))
        
        self.n = n 
        self.m = m 

    def get_init_obs(self):
        return self.init_obs
    
    def step(self, actions):
        obs = np.zeros((self.n*self.m, *self.shape), dtype=self.dtype)
        reward = np.zeros((self.n*self.m,), dtype=float)
        done = np.zeros((self.n*self.m,), dtype=bool)


        for i in range(self.n):
            self.outputPipe[i].send(
                actions[self.m*i:self.m*(i+1)]
            )
        
        for i in range(self.n):

            reward[self.m*i:self.m*(i+1)] = np.frombuffer(self.inputPipe[i].recv_bytes(), dtype=float)
            done[self.m*i:self.m*(i+1)] = np.frombuffer(self.inputPipe[i].recv_bytes(), dtype=bool)

            obs[self.m*i:self.m*(i+1)] = np.frombuffer(
                self.inputPipe[i].recv_bytes(), dtype=self.dtype).reshape((self.m, *self.shape))
        
        return obs, reward, done, {}

