import os
#os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import numpy as np
import envpool
import gym

import time

from V_TRACE import *
from model import *
import optax

from functools import partial
from jax.random import PRNGKey, split
import jax.numpy as jnp
import jax


import threading

def safe_print():
    real_print = print
    lock = threading.Lock()

    def aux(*args, **kargs):
        lock.acquire()
        real_print(*args, **kargs)
        lock.release()

    return aux

print = safe_print()


N = 3

actor = V_TRACE(
    ConvModel, (4, 84, 84), 6,
    1, N, jnp.array([0.99]),
    optax.adam(5e-4)
)
"""
actor = V_TRACE(
    MLP_MODEL, (128,), 6,
    1, N, jnp.array([0.99]),
    optax.adam(1e-3),
    E_coef=0.9
)
"""


params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)


agent.obs_process = jax.jit(lambda x : x / 255)


def work(actor, total_time, Q, num_envs=32, seed=42):
    rng = PRNGKey(seed)

    @jax.jit
    def get_action(rng, params, obs):
        rng1, rng2 = split(rng, num=2)
        logits, softmax = jax.tree_map(lambda t : t[0], actor.get_main_proba(params, obs[None]))
        action = jax.vmap(lambda s, r : jax.random.choice(r, a=actor.outDim, p=s))(softmax, jnp.array(split(rng, len(obs))))
        return action, logits, rng2

    global params
    global opti_state
    partial_tau  = PartialTau(N)
    current_r_sum = jnp.zeros((num_envs,))
    last_r_sum    = jnp.zeros((num_envs,))
    episode_len   = jnp.zeros((num_envs,))
    reward_curve  = []

    env = envpool.make("Pong-v5", env_type="gym", num_envs=num_envs)
    print(env.action_space)
    
    obs = env.reset()

    steps = 0
    start_time = time.time()

    while time.time() - start_time < total_time:

        action, logits, rng = jax.tree_map(np.array,  get_action(rng, params, obs))
        n_obs, reward, done, info = env.step(action)
        
        steps += 1
        tau = partial_tau.add_transition(obs, logits, action, reward, done, n_obs)
        obs = n_obs

        current_r_sum = reward + current_r_sum
        last_r_sum    = (1-done) * last_r_sum + done * current_r_sum
        episode_len   = (1 + episode_len) * (1-done)
        current_r_sum = current_r_sum * (1-done)

        if not tau is None: Q.put(tau)

        if steps % 100 == 0:
            print(end=f"\r{steps*num_envs}  {time.time()-start_time:.0f}  {np.mean(episode_len):.0f}  {np.mean(last_r_sum):.5f}  {np.exp(params[1]):.5f}   ")
            reward_curve.append(np.mean(last_r_sum))

    print()
    import matplotlib.pyplot as plt
    plt.plot(reward_curve)
    plt.show()

total_time = 30 * 60

import queue
Q = queue.Queue(10)

import threading
threads = [threading.Thread(target=work, args=(actor, total_time, Q)) for _ in range(2)]
[t.start() for t in threads]


while True:
    tau = Q.get
    opti_state, params, loss = actor.V_TRACE_step(opti_state, params, tau)


[t.join() for t in threads]