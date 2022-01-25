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

N = 3

actor = V_TRACE(
    ConvModel, (4, 84, 84), 6,
    1, N, jnp.array([0.99]),
    optax.adam(2e-4),
    E_coef=0.9
)

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)


import queue
Q = queue.Queue(128)

@jax.jit
def obs_process(obs):
    return obs / 255.0


def work(actor, total_time, Q, batch_size=32, num_envs=64, seed=42):
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
    total_reward = {}

    env = envpool.make("Pong-v5", env_type="gym", num_envs=num_envs, batch_size=batch_size)
    env.async_reset()


    steps = 0
    start_time = time.time()

    while time.time() - start_time < total_time:
        obs, reward, done, info = env.recv()

        action, logits, rng = jax.tree_map(np.array,  get_action(rng, params, obs))
        
        env.send(action, info['env_id'])
        steps += 1
        
        for i, ident in enumerate(info['env_id']):

            if not ident in partial_tau:
                partial_tau[ident]  = PartialTau(N)
                total_reward[ident] = 0.0

            else:
                p_obs, p_action, p_logits = last_state[ident]
                tau = partial_tau[ident].add_transition(p_obs, p_logits, p_action, reward[i], done[i], obs[i])
                if not tau is None: Q.put(tau)

            last_state[ident] = obs[i], action[i], logits[i]

            total_reward[ident] += reward[i]
            if done[i]:
                print(steps, int(time.time()-start_time), total_reward[ident])
                total_reward[ident] = 0.0


total_time = 3600

import threading
threads = [threading.Thread(target=work, args=(actor, total_time, Q)) for _ in range(1)]
[t.start() for t in threads]


steps = 0
wait_time = 0
start_time = time.time()
while time.time() - start_time < total_time:

    wait_time -= time.time()

    batch = []

    while len(batch) < 32:
        batch.append(Q.get())
    wait_time += time.time()

    batch = Tau(
        obs   =obs_process(jnp.array([[b.obs[i]    for b in batch] for i in range(N+1)])),
        reward=jnp.array([[b.reward[i] for b in batch] for i in range(N)]),
        done  =jnp.array([[b.done[i]   for b in batch] for i in range(N)]),
        action=jnp.array([[b.action[i] for b in batch] for i in range(N)]),
        logits=jnp.array([[b.logits[i] for b in batch] for i in range(N)])
    )

    opti_state, params, loss = actor.V_TRACE_step(opti_state, params, batch)
    steps += 1 

    if steps % 10 == 0: print(steps, int(time.time() - start_time), wait_time / (time.time() - start_time), np.exp(params[1]))


[t.join() for t in threads]