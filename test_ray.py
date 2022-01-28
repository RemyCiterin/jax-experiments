import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import vizdoomgym
import gym

import jax
from jax.random import PRNGKey, split

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
from wrapper_deepmind import *

class InvLazy(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return obs.__array__()

env_fn = lambda : InvLazy(wrap_deepmind(make_atari("BreakoutNoFrameskip-v4", show=False), frame_stack=True, scale=False, grayscale=True))
#env_fn = lambda : NormWrapper(DivReward(
#    Buffer(
#        SkipFrames(
#            Transpose(Doom(gym.make('VizdoomBasic-v0'))), 
#            number=4), 
#        number=4), 
#    100.0), 6.5, 21.5
#)
#env_fn = lambda : DivReward(gym.make("LunarLander-v2"), 10.0)

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

        obs = self.env.reset()
        print(self.env.action_space)

        @jax.jit
        def get_action(rng, params, obs):
            rng1, rng2 = split(rng, num=2)
            logits, softmax = jax.tree_map(lambda t : t[0, 0], self.agent.get_main_proba(params, obs[None, None]))
            action = jax.random.choice(rng1, a=self.agent.outDim, p=softmax)
            return action, logits, rng2

        rng = PRNGKey(42)

        while time.time() - start_time < total_time:

            action, logits, rng = from_jnp(get_action(rng, params, obs))

            n_obs, reward, done, _ = self.env.step(action)
            if done: n_obs = self.env.reset()
            r_sum += reward 

            tau = partial_tau.add_transition(obs, logits, action, np.clip(reward, -1, 1), done, n_obs)
            obs = n_obs
            steps += 1

            if not tau is None: buffer.put(tau)
            if steps % 100 == 0: params = from_np(ray.get(server.get_params.remote()))

            if done:
                print(steps, int(time.time()-start_time), str(r_sum)[:6])
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
from V_TRACE import *

N = 10
opti = optax.chain(
    optax.clip_by_global_norm(40.0),
    optax.rmsprop(5e-4, decay=0.99)
)

actor = V_TRACE(
    ConvModel, (4, 84, 84), 4,
    1, N, jnp.array([0.99]),
    opti=opti
)

actor.obs_process = jax.jit(lambda x : jax.numpy.transpose(x / 255, (0, 1, 4, 2, 3)))

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)

server = ParamsServer.remote(params)
buffer = ray.util.queue.Queue(128)

num_worker = 15
worker = [Worker.remote(env_fn, actor, N) for _ in range(num_worker)]
work = [w.work.remote(3600, buffer, server, i==0) for i, w in enumerate(worker)]


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

    batch = jax.tree_multimap(lambda *args: np.stack(args, axis=1), *batch)

    opti_state, params, loss = actor.V_TRACE_step(opti_state, params, batch, 
        H_target=1-(time.time()-start_time) / 14400
    )
    server.set_params.remote(from_jnp(params))
    steps += 1 

    if steps % 10 == 0: print(steps, int(time.time() - start_time), wait_time / (time.time() - start_time), np.exp(params[1]))

ray.get(work)
