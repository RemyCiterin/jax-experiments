# attention : très lent
from tensorboardX import SummaryWriter
from sync_env import VecEnv

import gym 

import numpy as np 

from V_TRACE import *
from model import *


import os
import jax 
from jax.random import PRNGKey, split
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

N_steps = 10

opti = optax.chain(
    optax.clip_by_global_norm(40.0),
    optax.rmsprop(5e-4, decay=0.99)
)

agent = V_TRACE(MLP_MODEL,
    inDim=(128,), outDim=6,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(6) / 5,
    opti=opti, use_Ftrace=False,
)

agent.obs_process = jax.jit(lambda x : x / 255)


params = agent.init_params(PRNGKey(42))
state = agent.init_state(params)

class Worker:
    def __init__(self, agent, learner_queue, env_fun, n, m):
        self.agent = agent

        self.queue = learner_queue

        self.env_list = VecEnv(env_fun, n, m)

        self.N = n * m

        self.partial_tau = PartialTau(N_steps)

    def work(self):

        @jax.jit
        def get_action(rng, params, obs):
            rng1, rng2 = split(rng, num=2)
            logits, softmax = jax.tree_map(lambda t : t[0], self.agent.get_main_proba(params, obs[None]))
            action = jax.vmap(lambda s, r : jax.random.choice(r, a=self.agent.outDim, p=s))(softmax, jnp.array(split(rng1, len(obs))))
            return action, logits, rng2

        rng = PRNGKey(42)
        #writer = SummaryWriter()
        print("worker start")
        obs = self.env_list.get_init_obs().astype(float)
        r_sum = [0.0 for _ in range(self.N)]
        steps = 0

        import time
        start_time = time.time()
        while True:
            
            action, logits, rng = get_action(rng, params, obs)

            n_obs, reward, done, _ = self.env_list.step(action)
            r_sum = [r+r_ for r, r_ in zip(r_sum, reward)]
            n_obs = n_obs.astype(float)
            

            tau = self.partial_tau.add_transition(obs, logits, action, reward, done, n_obs)
            if not tau is None: self.queue.put(tau)

            for i in range(self.N):
                if done[i]:
                    #if True: writer.add_scalar("reward", r_sum[i], steps*self.N)
                    print(steps*self.N, int(time.time()-start_time), r_sum[i])
                    #writer.flush()
                    r_sum[i] = 0
            
            obs = n_obs
            steps += 1

import queue 
learner_queue = queue.Queue(10)

import vizdoomgym
from Wrapper import *
worker = [
    Worker(agent, learner_queue, 
        #lambda : Buffer(SkipFrames(Transpose(Doom(gym.make("VizdoomBasic-v0"))), number=10), number=10),
        #lambda : Transpose(gym.make("MinAtar/Breakout-v0")), 
        lambda : gym.make("Pong-ram-v0"), 
        8, 4
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



while True:
    batch = learner_queue.get()

    #batch = jax.tree_multimap(lambda *args: np.stack(args, axis=1), *batch)

    state, params, loss = agent.V_TRACE_step(state, params, batch)



