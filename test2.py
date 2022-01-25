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

env_fn = lambda : NormWrapper(DivReward(
    Buffer(
        SkipFrames(
            Transpose(Doom(gym.make('VizdoomBasic-v0'))), 
            number=10), 
        number=10), 
    100.0), 6.5, 21.5
)
import time 
class Env_Vec:
    def __init__(self, env_fn, n):
        self.env = []
        for _ in range(n):
            self.env.append(env_fn())
            time.sleep(0.3)
    
    def reset(self):
        return np.array(tuple(env.reset() for env in self.env))
    
    def step(self, actions):
        obs_list, reward_list, done_list = [], [], []
        for env, action in zip(self.env, actions):
            obs, reward, done, _ = env.step(action)
            if done: obs = env.reset()
            reward_list.append(reward)
            done_list.append(done)
            obs_list.append(obs)
        
        return np.array(obs_list), np.array(reward_list), np.array(done_list), {}

from vec_env import *
env = VecEnv(env_fn, 12, 4, render=True)
#env = Env_Vec(env_fn, 16)

from ETD import *
from model import *
import optax 

N = 1
actor = ETD(
    ConvModel, (10, 48, 64), 3,
    1, N, jnp.array([0.99]),
    optax.adam(1e-3),
    E_coef=0.05
)

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)

partial_tau = PartialTau(N)

#obs = env.reset()
obs = env.get_init_obs()

step = 0
print(obs.shape)
current_r_sum = np.zeros((len(obs),))
last_r_sum    = np.zeros((len(obs),))

results = []


start_time = time.time()
import matplotlib.pyplot as plt
while time.time()-start_time < 60*30:
    logits, softmax = jax.tree_map(
        lambda t : np.array(t[0]), 
        actor.get_main_proba(params, obs[None, ...])
    )

    action = np.array([np.random.choice(actor.outDim, p=s / np.sum(s)) for s in softmax], dtype=int)
    
    n_obs, reward, done, _ = env.step(action)

    current_r_sum = reward + current_r_sum
    last_r_sum    = (1-done) * last_r_sum + done * current_r_sum
    current_r_sum = current_r_sum * (1-done)

    tau = partial_tau.add_transition(obs, logits, action, reward, done, n_obs)
    obs = n_obs
    step += 1
    
    if not tau is None:
        batch = Tau(
            obs = jnp.array(tau.obs), 
            done = jnp.array(tau.done), 
            reward = jnp.array(tau.reward),
            action = jnp.array(tau.action),
            logits = jnp.array(tau.logits)
        )

        opti_state, params, loss = actor.ETD_step(opti_state, params, batch)

        print(str(loss)[:5], step, int(100*np.mean(last_r_sum)), int(time.time()-start_time))
        results.append(100*np.mean(last_r_sum))
    
    #env.env[0].render()
plt.plot(results)
plt.show()