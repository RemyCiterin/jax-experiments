from torch.utils.tensorboard import SummaryWriter
from vec_env import VecEnv

import gym 

import numpy as np 

from ETD import * 
from model import *


import os
import jax 
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

N_steps = 5

opti = optax.chain(
    optax.adam(learning_rate=2e-4)
)

agent = ETD(MLP_MODEL,
    inDim=(128,), outDim=6,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(6) / 5,
    opti=opti, E_coef=0.01,
    use_Ftrace=False,
)

"""
agent = ETD(LittleConvModel,
    inDim=(4, 10, 10), outDim=6,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(6) / 5,
    opti=opti, E_coef=0.05,
    use_Ftrace=True,
)
"""
"""
agent = ETD(ConvModel, 
    inDim=(10, 64, 48), outDim=3,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(3)/5,
    opti=opti, E_coef=0.05,
    use_Ftrace=False,
)
"""
params = agent.init_params(PRNGKey(42))
state = agent.init_state(params)

class Worker:
    def __init__(self, agent, learner_queue, env_fun, n, m):
        self.agent = agent

        self.queue = learner_queue

        self.env_list = VecEnv(env_fun, n, m)

        self.N = n * m

        self.partial_tau = [
            PartialTau(N_steps) for _ in range(n*m)
        ]

    def work(self):
        writer = SummaryWriter()
        print("worker start")
        obs = self.env_list.get_init_obs().astype(float)
        r_sum = [0.0 for _ in range(self.N)]
        steps = 0

        while True:
            
            logits, softmax = jax.tree_map(
                lambda t : np.array(t)[0],
                self.agent.get_main_proba(params, obs[None, ...])
            )

            action = np.array([
                np.random.choice(self.agent.outDim, p=s / np.sum(s)) for s in softmax
            ])

            n_obs, reward, done, _ = self.env_list.step(action)
            r_sum = [r+r_ for r, r_ in zip(r_sum, reward)]
            n_obs = n_obs.astype(float)
            

            for i in range(self.N):
                tau = self.partial_tau[i].add_transition(obs[i], logits[i], action[i], reward[i], done[i], n_obs[i])
                if not tau is None: self.queue.put(tau)

                if done[i]:
                    if True: writer.add_scalar("reward", r_sum[i], steps*self.N)
                    print(steps*self.N, r_sum[i])
                    writer.flush()
                    r_sum[i] = 0
            
            obs = n_obs
            steps += 1

import queue 
learner_queue = queue.Queue(128)

import vizdoomgym
from Wrapper import *
worker = [
    Worker(agent, learner_queue, 
        #lambda : Buffer(SkipFrames(Transpose(Doom(gym.make("VizdoomBasic-v0"))), number=10), number=10),
        #lambda : Transpose(gym.make("MinAtar/Breakout-v0")), 
        lambda : FromUInt8(gym.make("Pong-ram-v0")), 
        8, 8
    )
    for _ in range(2)
]

import threading
[threading.Thread(target=w.work).start() for w in worker]


import os
import time
import Buffer
print("learner start")
start_time = time.time()
#buffer = Buffer.Buffer(200)
step = 0

counter = 0
while True:
    batch = []
    counter -= time.time()
    while len(batch) < 32: batch.append(learner_queue.get())
    counter += time.time()

    #buffer.add_tau_list(batch)
    #batch = batch + buffer.sample_batch(24)[0]
    #if buffer.size < 128: continue

    batch = Tau(
        obs   =jnp.array([[b.obs[i]    for b in batch] for i in range(2*N_steps)]), 
        reward=jnp.array([[b.reward[i] for b in batch] for i in range(2*N_steps-1)]), 
        done  =jnp.array([[b.done[i]   for b in batch] for i in range(2*N_steps-1)]), 
        action=jnp.array([[b.action[i] for b in batch] for i in range(2*N_steps-1)]), 
        logits=jnp.array([[b.logits[i] for b in batch] for i in range(2*N_steps-1)])
    )

    state, params, loss = agent.ETD_step(state, params, batch)

    step += 1
    if step % 10 == 0:
        tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
        #writer.add_scalar("memory", used_m / tot_m, step)
    
        #print(end="\rsteps = {}   loss = {}     {}     {}   ".format(
        #    step, np.mean(loss), counter / (time.time()-start_time), used_m / tot_m
        #))



