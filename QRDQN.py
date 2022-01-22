from jax.random import PRNGKey, split 
import jax.numpy as jnp
from jax import vmap
import optax 
import jax 

import haiku as hk 
import numpy as np 

from collections import namedtuple
from typing import Callable, NamedTuple
import functools

sg = jax.lax.stop_gradient


Tau = namedtuple('Tau', ["obs", "reward", "gamma", "action", "n_obs"])

class QRDQN:
    def __init__(self, core:Callable[[int, int], hk.Module], inDim, outDim:int, 
            n_atoms:int=32, opti=optax.adam(2e-4)):

        self._init_fn, self.apply_fn = hk.without_apply_rng(
            hk.transform(lambda x : core(outDim * n_atoms, num_heads=1)(x[None]).logits[0, 0])
        )

        self.opti = opti

        self.n_atoms = n_atoms 
        self.outDim = outDim 
        self.inDim = inDim 
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def init_params(self, key):
        return self._init_fn(key, jnp.zeros(self.inDim)[None])
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def init_state(self, params):
        return self.opti.init(params)

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs):
        #obs : batch of observations

        Z = self.apply_fn(params, obs).reshape(-1, self.outDim, self.n_atoms)

        return jnp.argmax(jnp.mean(Z, axis=-1), axis=-1)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def QRDQN_loss(self, params, target, tau:Tau, weight=1):
        
        # apply_fn need batch of batch
        Z = self.apply_fn(params, tau.obs).reshape(-1, self.outDim, self.n_atoms)

        nZ1 = self.apply_fn(params, tau.n_obs).reshape(-1, self.outDim, self.n_atoms)

        nA = jnp.argmax(jnp.mean(nZ1, axis=-1), axis=-1)

        nZ2 = self.apply_fn(target, tau.n_obs).reshape(-1, self.outDim, self.n_atoms)
        
        Z  = jnp.sum(Z   * jax.nn.one_hot(tau.action, self.outDim)[..., None], axis=1)
        nZ = jnp.sum(nZ2 * jax.nn.one_hot(nA,         self.outDim)[..., None], axis=1)

        tZ = tau.reward[..., None] + tau.gamma[..., None] * sg(nZ)

        U = tZ[::, None, ::] - Z[::, ::, None]

        t = (jnp.arange(self.n_atoms) + 0.5) / self.n_atoms
        
        L = 0.5 * jnp.less(jnp.abs(U), 1.0) * U ** 2 + (jnp.abs(U) - 0.5) * jnp.greater_equal(jnp.abs(U), 1.0)

        rho = jnp.abs(t[None, ::, None] - jnp.less(U, 0.0)) * L

        prio = jnp.mean(jnp.sum(rho, axis=-1), axis=-1)

        return jnp.mean(weight * prio), prio
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def QRDQN_step(self, state, params, target, tau, weight=1.0):
        (loss, prio), grad = jax.value_and_grad(self.QRDQN_loss, has_aux=True)(params, target, tau, weight)

        updates, state = self.opti.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss, prio

class PartialTau:
    def __init__(self, n_steps, gamma=0.99):
        self.n_steps = n_steps 
        self.gamma   = gamma

        self.buffer = []
    
    def add_transition(self, obs, action, reward, done, n_obs):
        
        self.buffer.append(Tau(obs, reward, (1-done), action, n_obs))

        if len(self.buffer) == self.n_steps:
            reward, gamma = 0, 1

            for b in self.buffer:
                reward += gamma * b.reward
                gamma *= self.gamma * b.gamma

            tau = Tau(
                obs=self.buffer[0].obs, reward=reward, 
                gamma=gamma, action=self.buffer[0].action, n_obs=n_obs
            )

            self.buffer = self.buffer[1:]
            return tau
        
        return None 

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from model import *

    from Buffer import Buffer

    opti = optax.chain(
        #optax.clip_by_global_norm(1.0), 
        optax.adam(learning_rate=2e-4)
    )

    n = 5

    actor = QRDQN(ConvModel, (10, 48, 64), 3)

    params = actor.init_params(PRNGKey(42))
    target = actor.init_params(PRNGKey(43))
    state = actor.init_state(params)

    n, m = 12, 2
    ALPHA = 0.6
    BETA = 0.4

    import gym 
    import Wrapper 
    import vizdoomgym
    from vec_env import VecEnv
    #env = VecEnv(lambda : Wrapper.Frameskip(Wrapper.Wrapper(gym.make("PongNoFrameskip-v0"))), n, m)
    env = VecEnv(lambda : Wrapper.Buffer(Wrapper.SkipFrames(Wrapper.Transpose(Wrapper.Doom(gym.make("VizdoomBasic-v0"))), number=10), number=10), n, m)
    obs = env.get_init_obs().astype(float)
    obs = (obs - np.mean(obs)) / np.std(obs)
    r_sum = np.zeros([n*m])
    step = 0

    import Buffer
    buffer = Buffer.Buffer(20000)
    partial_tau = PartialTau(n)

    writer = SummaryWriter()

    eps = np.random.uniform(size=(n*m,)) / 2.0

    while True:
        action = np.array(actor.get_action(params, obs))

        #eps = max(0.05, 1 - step/2000)
        U = np.random.uniform(size=n * m)
        action = action * (U > eps) + np.random.choice(actor.outDim, size=n * m) * (U <= eps)

        n_obs, reward, done, _ = env.step(action)
        n_obs = n_obs.astype(float)
        r_sum += reward

        n_obs = (n_obs - np.mean(n_obs)) / np.std(n_obs)

        for i in range(n*m):
            if done[i]: 
                if eps[i] < 0.1:
                    print(step*n*m, r_sum[i])
                    writer.add_scalar("reward", r_sum[i], step*n*m)
                    writer.flush()
                r_sum[i] = 0

        tau = partial_tau.add_transition(obs, action, reward, done, n_obs)

        if not tau is None: 
            #_, prio = actor.QRDQN_loss(params, target, jax.tree_map(jnp.array, tau))
            for i in range(n*m): buffer.add_tau(jax.tree_map(lambda t : t[i] ,tau), prio=None)#np.array(prio[i]) ** ALPHA)

        if buffer.size > 1000:
            for _ in range(8):
                batch, index, weight, mini = buffer.sample_batch(32)
                weight = np.array([(mini / w) ** BETA for w in weight])

                batch = Tau(
                    obs   =jnp.array([b.obs    for b in batch]), 
                    reward=jnp.array([b.reward for b in batch]), 
                    gamma =jnp.array([b.gamma  for b in batch]), 
                    action=jnp.array([b.action for b in batch]), 
                    n_obs =jnp.array([b.n_obs  for b in batch])
                )

                state, params, loss, new_prio = actor.QRDQN_step(state, params, target, batch)
                buffer.update(index, np.array(new_prio) ** ALPHA)
        
        if step % 100 == 0:
            target = params
        
        step += 1
        obs = n_obs

