import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import vizdoomgym
import gym

from Wrapper import *

class DivReward(gym.RewardWrapper):
    def __init__(self, env, x) -> None:
        super().__init__(env)
        self.x = x
    
    def reward(self, rew):
        return rew / self.x

class NormWrapper(gym.ObservationWrapper):
    def __init__(self, env, mu, std) -> None:
        super().__init__(env)
        self.std = std 
        self.mu = mu
    
    def observation(self, observation):
        return (observation - self.mu) / self.std

#env_fn = lambda : NormWrapper(DivReward(
#    Buffer(
#        SkipFrames(
#            Transpose(Doom(gym.make('VizdoomBasic-v0'))), 
#            number=10), 
#        number=10), 
#    100.0), 6.5, 21.5
#)
env_fn = lambda : DivReward(gym.make("LunarLander-v2"), 10.0)

import time 

import ray 

ray.init()

import ray.util.queue 

import numpy as np 
import jax.numpy as jnp 
import jax 

from functools import partial

from_np = partial(jax.tree_map, jnp.array)
from_jnp = partial(jax.tree_map, np.array)

@ray.remote 
class Worker:
    def __init__(self, env_fn, agent, N, backend="cpu"):
        os.environ['JAX_PLATFORM_NAME'] = backend
        import vizdoomgym
        self.env = env_fn()
        self.agent = agent 
        self.N = N 

    def work(self, total_time, buffer, server, render=False):
        params = from_np(ray.get(server.get_params.remote()))

        r_sum = 0
        steps = 0
        start_time = time.time()
        partial_tau = PartialTau(self.N)

        entropy, len_ep = 0, 0

        obs = self.env.reset()

        while time.time() - start_time < total_time:

            logits, softmax = jax.tree_map(
                lambda t : np.array(t[0, 0]), 
                actor.get_main_proba(params, obs[None, None])
            )

            action = np.random.choice(actor.outDim, p=softmax / np.sum(softmax))

            entropy += -np.sum(np.log(softmax) * softmax)
            len_ep  += 1

            n_obs, reward, done, _ = self.env.step(action)
            if done: n_obs = self.env.reset()
            r_sum += reward 

            tau = partial_tau.add_transition(obs, logits, action, reward, done, n_obs)
            obs = n_obs
            steps += 1

            if not tau is None: buffer.put(tau)
            if steps % 100 == 0: params = from_np(ray.get(server.get_params.remote()))

            if done:
                print(steps, int(time.time()-start_time), str(r_sum)[:6], str(entropy / len_ep)[:6])
                entropy, len_ep = 0, 0
                r_sum = 0

            if render: self.env.render()

@ray.remote
class ParamsServer:
    def __init__(self, params):
        self.params = from_jnp(params)

    def get_params(self):
        return self.params 

    def set_params(self, params):
        self.params = params 

from model import * 
from ETD import *

N = 1
actor = ETD(
    MLP_MODEL, (8,), 4,
    1, N, jnp.array([0.99]),
    optax.adam(1e-3),
    E_coef=0.98
)

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)

server = ParamsServer.remote(params)
buffer = ray.util.queue.Queue(128)

num_worker = 10 
worker = [Worker.remote(env_fn, actor, N) for _ in range(num_worker)]
work = [w.work.remote(360, buffer, server, i==0) for i, w in enumerate(worker)]


steps = 0
wait_time = 0
start_time = time.time()
while True:

    wait_time -= time.time()
    while True:
        try:
            batch = buffer.get_nowait_batch(32); break
        except ray.util.queue.Empty: pass
    wait_time += time.time()

    batch = Tau(
        obs   =jnp.array([[b.obs[i]    for b in batch] for i in range(2*N)]), 
        reward=jnp.array([[b.reward[i] for b in batch] for i in range(2*N-1)]), 
        done  =jnp.array([[b.done[i]   for b in batch] for i in range(2*N-1)]), 
        action=jnp.array([[b.action[i] for b in batch] for i in range(2*N-1)]), 
        logits=jnp.array([[b.logits[i] for b in batch] for i in range(2*N-1)])
    )

    opti_state, params, loss = actor.ETD_step(opti_state, params, batch)
    server.set_params.remote(from_jnp(params))
    steps += 1 

    if steps % 10 == 0: print(steps, int(time.time() - start_time), wait_time / (time.time() - start_time), np.exp(params[1]))

ray.get(work)
