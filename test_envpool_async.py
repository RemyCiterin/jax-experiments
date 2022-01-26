import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import numpy as np 
import envpool 
import gym 

import time

from V_TRACE import *
from model import *
import numpy as np
import optax 

from functools import partial
from jax.random import PRNGKey, split
import jax.numpy as jnp
import jax

from typing import NamedTuple

N = 5

actor = V_TRACE(
    ConvModel, (4, 84, 84), 6,
    1, N, jnp.array([0.99]),
    optax.adam(5e-4),
    E_coef=0.9
)

"""

N = 10

actor = V_TRACE(
    ConvModel, (4, 84, 84), 5,
    1, N, jnp.array([0.99]),
    optax.adam(5e-4),
    E_coef=0.9
)

"""

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)


import queue
Q = queue.Queue(64)


@jax.jit
def obs_process(obs):
    return obs / 255.0


def work(actor, total_time, Q, batch_size=32, num_envs=96, seed=42):
    rng = PRNGKey(seed)

    @jax.jit
    def get_action(rng, params, obs):
        rng1, rng2 = split(rng, num=2)
        logits, softmax = jax.tree_map(lambda t : t[0], actor.get_main_proba(params, obs_process(obs[None])))
        action = jax.vmap(lambda s, r : jax.random.choice(r, a=actor.outDim, p=s))(softmax, jnp.array(split(rng, len(obs))))
        return action, logits, rng2


    global params
    partial_tau  = {}
    last_state   = {}

    current_r_sum = jnp.zeros((num_envs,))
    last_r_sum    = jnp.zeros((num_envs,))

    @partial(jax.jit, backend="cpu")
    def update_recoder(idx, reward, done, c_sum, l_sum):
        return (
            c_sum.at[idx].set((c_sum.at[idx].get() + reward) * (1 - done)), 
            l_sum.at[idx].set((c_sum.at[idx].get() + reward) * done + l_sum.at[idx].get() * (1-done))
        )

    env = envpool.make("Pong-v5", env_type="gym", num_envs=num_envs, batch_size=batch_size)
    env.async_reset()


    steps = 0
    start_time = time.time()

    reward_curve = []

    while time.time() - start_time < total_time:
        obs, reward, done, info = env.recv()

        action, logits, rng = jax.tree_map(np.array,  get_action(rng, params, obs))
        
        env.send(action, info['env_id'])
        steps += 1

        current_r_sum, last_r_sum = update_recoder(info['env_id'], reward, done, current_r_sum, last_r_sum)
        
        
        for i, ident in enumerate(info['env_id']):

            if not ident in partial_tau:
                partial_tau[ident] = PartialTau(N)

            if ident in last_state:
                p_obs, p_action, p_logits = last_state[ident]
                tau = partial_tau[ident].add_transition(p_obs, p_logits, p_action, reward[i], done[i], obs[i])
                if not tau is None: Q.put(tau)
                

            last_state[ident] = obs[i], action[i], logits[i]


        if steps % 100 == 0:
            print(end=f"\r{steps*batch_size}  {time.time()-start_time:.0f}  {np.mean(last_r_sum):.6f}  ")
            reward_curve.append(np.mean(last_r_sum))

    import matplotlib.pyplot as plt
    plt.plot(reward_curve)
    plt.show()


total_time = 60 * 30

import threading
thread = threading.Thread(target=work, args=(actor, total_time, Q))
thread.start()

while True:

    batch = []
    while len(batch) < 32:
        batch.append(Q.get())

    batch = Tau(
        obs   =obs_process(jnp.array([[b.obs[i] for b in batch] for i in range(N+1)])),
        reward=jnp.array([[b.reward[i] for b in batch] for i in range(N)]),
        done  =jnp.array([[b.done[i]   for b in batch] for i in range(N)]),
        action=jnp.array([[b.action[i] for b in batch] for i in range(N)]),
        logits=jnp.array([[b.logits[i] for b in batch] for i in range(N)])
    )

    opti_state, params, loss = actor.V_TRACE_step(opti_state, params, batch)

thread.join()