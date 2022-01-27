from torch.utils.tensorboard import SummaryWriter
from sync_env import VecEnv

import gym 

import numpy as np 

from QRDQN import *
from model import *

from queue import Queue
from Buffer import Buffer 

import os
import jax 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.20'

N_steps = 10

opti = optax.chain(
    optax.adam(learning_rate=1e-3)
)

agent = QRDQN(MLP_MODEL, (4,), 2, opti=opti)

params = agent.init_params(PRNGKey(42))
target = agent.init_params(PRNGKey(43))
state = agent.init_state(params)

ALPHA = 0.6
BETA  = 0.4

class Worker:
    def __init__(self, agent, buffer : Buffer, env_fun, n, m):
        self.agent = agent

        self.buffer = buffer

        self.env_list = VecEnv(env_fun, n, m)

        self.N = n * m

        self.partial_tau = PartialTau(N_steps)

    def work(self):
        writer = SummaryWriter()
        print("worker start")
        obs = self.env_list.get_init_obs().astype(float)
        r_sum = [0.0 for _ in range(self.N)]
        steps = 0

        eps = np.random.uniform(size=self.N) / 2.0
        

        print(eps)

        while True:
            
            action = self.agent.get_action(params, obs)

            U = np.random.uniform(size=self.N)
            action = action * (U > eps) + np.random.choice(self.agent.outDim, size=self.N) * (U <= eps)

            n_obs, reward, done, _ = self.env_list.step(action)
            r_sum = [r+r_ for r, r_ in zip(r_sum, reward)]
            n_obs = n_obs.astype(float)
            
            tau = self.partial_tau.add_transition(obs, action, reward, done, n_obs)

            if not tau is None:
                _, prio = agent.QRDQN_loss(params, target, jax.tree_map(np.array, tau))
                for i in range(self.N):
                    self.buffer.add_tau(
                        jax.tree_map(lambda t :t[i], tau), np.array(prio[i]) ** ALPHA
                    )

            for i in range(self.N):
                if done[i]:
                    if eps[i] < 0.1: writer.add_scalar("reward", r_sum[i], steps*self.N)
                    if eps[i] < 0.1: print(steps*self.N, r_sum[i])
                    writer.flush()
                    r_sum[i] = 0
            
            obs = n_obs
            steps += 1

buffer = Buffer(10000)

import vizdoomgym
from Wrapper import *
worker = [
    Worker(agent, buffer,
        #lambda : Buffer(SkipFrames(Transpose(Doom(gym.make("VizdoomBasic-v0"))), number=10), number=10),
        #lambda : Transpose(gym.make("MinAtar/Breakout-v0")), 
        #lambda : FromUInt8(gym.make("Pong-ram-v0")), 
        lambda : gym.make("CartPole-v1"),
        8, 8
    )
    for _ in range(1)
]

import threading
[threading.Thread(target=w.work).start() for w in worker]


import os
import time
print("learner start")
start_time = time.time()

step = 0

while True:
    
    batch, index, weight, m = buffer.sample_batch(32)
    if buffer.size < 1000:
        time.sleep(1)
        continue
    

    batch = Tau(
        obs   =jnp.array([b.obs    for b in batch]),
        reward=jnp.array([b.reward for b in batch]),
        gamma =jnp.array([b.gamma  for b in batch]),
        action=jnp.array([b.action for b in batch]),
        n_obs =jnp.array([b.n_obs  for b in batch])
    )

    weight = np.array([(m / w) ** BETA for w in weight])

    state, params, loss, new_prio = agent.QRDQN_step(state, params, target, batch, weight)

    buffer.update(index, np.array(new_prio) ** ALPHA)

    if step % 100 == 0:
        target = params
        #writer.flush()

    step += 1
    if step % 10 == 0:
        pass#tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        #writer.add_scalar("memory", used_m / tot_m, step)
    
        #print(end="\rsteps = {}   loss = {}     {}     {}   ".format(
        #    step, np.mean(loss), time.time()-start_time, used_m / tot_m
        #))



