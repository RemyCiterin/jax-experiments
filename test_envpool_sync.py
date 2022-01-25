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

N = 3

actor = V_TRACE(
    ConvModel, (4, 84, 84), 6,
    1, N, jnp.array([0.99]),
    optax.adam(2e-4),
    E_coef=0.9
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



def work(actor, total_time, num_envs=64, seed=42):
    rng = PRNGKey(seed)

    @jax.jit
    def obs_process(obs):
        return obs / 255.0

    @jax.jit
    def get_action(rng, params, obs):
        rng1, rng2 = split(rng, num=2)
        logits, softmax = jax.tree_map(lambda t : t[0], actor.get_main_proba(params, obs_process(obs[None])))
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

        
        
        if not tau is None:
            tau = Tau( 
                done = jnp.array(tau.done),
                reward = jnp.array(tau.reward),
                action = jnp.array(tau.action),
                logits = jnp.array(tau.logits),
                obs = obs_process(jnp.array(tau.obs)),
            )

            opti_state, params, loss = actor.V_TRACE_step(opti_state, params, tau)
            print(end=f"\r{steps*num_envs}  {time.time()-start_time:.0f}  {np.mean(episode_len):.0f}  {np.mean(last_r_sum):.5f}  {np.exp(params[1]):.5f}   ")
            reward_curve.append(np.mean(last_r_sum))

    print()
    return reward_curve

 
results = work(actor, 3600)

import matplotlib.pyplot as plt
plt.plot(results)
plt.show()