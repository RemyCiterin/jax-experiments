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
import pickle

import tensorboardX

writer = tensorboardX.SummaryWriter()

N = 10

optim = optax.chain(
    optax.clip_by_global_norm(40.0),
    optax.rmsprop(5e-4, decay=0.99)
)

actor = V_TRACE(
    ConvModel, (12, 84, 84), 6,
    #3, N, jnp.array([0.99, 0.988, 0.985]),
    1, N, jnp.array([0.99]),
    trust_region=None, 
    use_Ftrace=False,
    DKL_target=None,
    opti=optim,
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

env_id = "DefendTheCenter-v1"#"Breakout-v5"

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)


import queue
Q = queue.Queue(64)

actor.obs_process = jax.jit(lambda obs : obs / 255)

def work(actor, total_time, Q, batch_size=32, num_envs=64, seed=42):
    rng = PRNGKey(seed)

    @jax.jit
    def get_action(rng, params, obs):
        rng1, rng2 = split(rng, num=2)
        logits, softmax = jax.tree_map(lambda t : t[0], actor.get_main_proba(params, obs[None]))
        action = jax.vmap(lambda s, r : jax.random.choice(r, a=actor.outDim, p=s))(softmax, jnp.array(split(rng1, len(obs))))
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

    #env = envpool.make("Basic-v1", env_type="gym", num_envs=num_envs, batch_size=batch_size, use_combined_action=True)
    env = envpool.make(env_id, env_type="gym", num_envs=num_envs, batch_size=batch_size, use_combined_action=True)
    print(env.action_space)
    env.async_reset()


    steps = 0
    start_time = time.time()

    reward_curve = []

    while time.time() - start_time < total_time:
        obs, reward, done, info = env.recv()

        action, logits, rng = jax.tree_map(np.asarray,  get_action(rng, params, obs))
        
        env.send(action.astype(np.int64), info['env_id'])
        steps += 1

        current_r_sum, last_r_sum = update_recoder(info['env_id'], reward, done, current_r_sum, last_r_sum)
        
        
        for i, ident in enumerate(info['env_id']):

            if not ident in partial_tau:
                partial_tau[ident] = PartialTau(N, use_ETD=False)

            if ident in last_state:
                p_obs, p_action, p_logits = last_state[ident]
                tau = partial_tau[ident].add_transition(p_obs, p_logits, p_action, np.clip(reward[i], -1, 1), done[i], obs[i])
                if not tau is None: Q.put(tau)
                

            last_state[ident] = obs[i], action[i], logits[i]


        if steps % 100 == 0:
            print(end=f"\r{steps*batch_size}  {time.time()-start_time:.0f}  {np.mean(last_r_sum):.6f}  {np.exp(params[1]):.6f}   {np.exp(params[2]):.6f}   ")
            writer.add_scalar(env_id, np.mean(last_r_sum), steps*batch_size)
            reward_curve.append(np.mean(last_r_sum))
            writer.flush()

        if steps % 10000 == 0:
            file = open("MODEL/" + env_id + "_buffer_" + str(steps) + ".pickle", "wb")
            pickle.dump(params, file)
            file.close()


    import matplotlib.pyplot as plt
    plt.plot(reward_curve)
    plt.show()

total_time = 60 * 30

import threading
thread = threading.Thread(target=work, args=(actor, total_time, Q))
thread.start()


from Buffer import *
buffer = Buffer(10000)

start_time = time.time()

while True:
    batch = []
    while len(batch) < 32:
        batch.append(Q.get())

    #buffer.add_tau_list(batch)
    #if buffer.size < 1024: continue
    #batch += buffer.sample_batch(24)[0]


    batch = jax.tree_multimap(lambda *args: np.stack(args, axis=1), *batch)

    opti_state, params, loss = actor.V_TRACE_step(opti_state, params, batch, H_target=1-(time.time()-start_time)/(60*120))

thread.join()