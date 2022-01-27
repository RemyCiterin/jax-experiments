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


Tau = namedtuple('Tau', ["obs", "reward", "done", "action", "logits"])


class V_TRACE(object):
    def __init__(self, core:Callable[[int], hk.Module], inDim, outDim:int, num_heads:int, 
            trajectory_n, gamma, opti=optax.adam(2e-4), trust_region=None, use_Ftrace=False, 
            init_H_coef=0.01, init_DKL_coef=5.0, DKL_target=None):

        self._init_fn, self.apply_fn = hk.without_apply_rng(
            hk.transform(lambda x : core(outDim, num_heads=num_heads)(x))
        )

        self.opti = opti

        self.init_DKL_coef = init_DKL_coef
        self.trust_region  = trust_region
        self.init_H_coef   = init_H_coef
        self.use_Ftrace    = use_Ftrace
        self.DKL_target    = DKL_target
        self.n      = trajectory_n
        self.N      = num_heads
        self.outDim = outDim 
        self.inDim  = inDim 
        self.gamma  = gamma

    @functools.partial(jax.jit, static_argnums=(0,))
    def obs_process(self, obs):
        return obs
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def init_params(self, key):
        return (self._init_fn(key, jnp.zeros(self.inDim)[None, None, ...]), jnp.log(self.init_H_coef), jnp.log(self.init_DKL_coef))
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def init_state(self, params):
        return self.opti.init(params)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_main_proba(self, params, obs):
        logits = self.apply_fn(params[0], self.obs_process(obs)).logits[0]
        return logits, jax.nn.softmax(logits)
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def V_TRACE_loss(self, params, tau:Tau, H_target:jnp.ndarray):

        assert len(tau.obs)    >= self.n+1
        assert len(tau.action) >= self.n
        assert len(tau.logits) >= self.n
        assert len(tau.reward) >= self.n
        assert len(tau.done)   >= self.n

        result = self.apply_fn(params[0], self.obs_process(tau.obs))

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.logits.at[:self.n].get()) * 
            jax.nn.one_hot(tau.action.at[:self.n].get(), self.outDim),
        axis=-1)

        def get_loss(value, logits, gamma):
            gamma = gamma * (1-tau.done.at[:self.n].get())

            ln_pi = jnp.sum(
                jax.nn.log_softmax(logits.at[:self.n].get()) * 
                jax.nn.one_hot(tau.action.at[:self.n].get(), self.outDim),
            axis=-1)

            RHO = sg(
                jnp.minimum(1, jnp.exp(ln_pi - ln_mu))
            )

            if not self.trust_region is None:
                Dkl = jnp.sum(
                    jax.nn.softmax(logits.at[:self.n].get()) * (
                        jax.nn.log_softmax(logits.at[:self.n].get()) - 
                        jax.nn.log_softmax(tau.logits.at[:self.n].get())
                ), axis=-1)

                RHO = RHO * jnp.less(Dkl, self.trust_region)

            delta = RHO * sg(
                tau.reward.at[:self.n].get() + gamma * 
                value.at[1:self.n+1].get() - value.at[:self.n].get()
            )

            
            E = vmap(lambda t : jnp.sum(vmap(lambda i :
                
                jnp.greater_equal(i, t) * delta.at[i].get() * jnp.prod(vmap(lambda j : 
                    gamma.at[j].get() * RHO.at[j].get() * jnp.less(j, i) * jnp.greater_equal(j, t) +
                    1 - jnp.less(j, i) * jnp.greater_equal(j, t)
                )(jnp.arange(self.n)), axis=0) #jnp.arange(t, i)
            
            )(jnp.arange(self.n)), axis=0))(jnp.arange(self.n))

            V = sg(value).at[:self.n+1].get().at[:self.n].add(E)

            lossV = (value.at[:self.n].get() - V.at[:self.n].get()) ** 2

            lossP = -ln_pi * RHO * sg(
                tau.reward.at[:self.n].get() + gamma.at[:self.n].get() * V.at[1:].get() - value.at[:self.n].get()
            )

            entropy = - jnp.sum(
                jax.nn.softmax(logits.at[:self.n].get()) *
                jax.nn.log_softmax(logits.at[:self.n].get()), 
            axis=-1)

            lossE = - jax.lax.stop_gradient(jnp.exp(params[1])) * entropy + \
                jnp.exp(params[1]) * jax.lax.stop_gradient(entropy - H_target * jnp.log(self.outDim))

            if not self.DKL_target is None:
                Dkl = jnp.sum(
                    jax.nn.softmax(logits.at[:self.n].get()) * (
                        jax.nn.log_softmax(logits.at[:self.n].get()) - 
                        jax.nn.log_softmax(tau.logits.at[:self.n].get())
                ), axis=-1)

                lossP = lossP + jnp.exp(params[2]) * sg(self.DKL_target - Dkl) + sg(jnp.exp(params[2])) * Dkl

            return lossV + lossP + lossE
        
        return jnp.mean(jax.vmap(get_loss)(result.value, result.logits, self.gamma))

    @functools.partial(jax.jit, static_argnums=(0,))
    def V_TRACE_step(self, state, params, tau, H_target=0.9):
        loss, grad = jax.value_and_grad(self.V_TRACE_loss)(params, tau, H_target)

        updates, state = self.opti.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss


    @functools.partial(jax.jit, static_argnums=(0,))
    def PG_loss(self, params, tau:Tau, H_target:jnp.ndarray):

        assert len(tau.obs)    >= self.n+1
        assert len(tau.action) >= self.n
        assert len(tau.logits) >= self.n
        assert len(tau.reward) >= self.n
        assert len(tau.done)   >= self.n

        result = self.apply_fn(params[0], self.obs_process(tau.obs))

        def get_loss(value, logits, gamma):
            gamma = gamma * (1-tau.done.at[:self.n].get())

            ln_pi = jnp.sum(
                jax.nn.log_softmax(logits.at[:self.n].get()) * 
                jax.nn.one_hot(tau.action.at[:self.n].get(), self.outDim),
            axis=-1)

            delta = sg(tau.reward.at[:self.n].get() + gamma * value.at[1:self.n+1].get()) - value.at[:self.n].get()

            lossV = delta ** 2

            lossP = -ln_pi * sg(delta)

            entropy = - jnp.sum(
                jax.nn.softmax(logits.at[:self.n].get()) *
                jax.nn.log_softmax(logits.at[:self.n].get()), 
            axis=-1)

            lossE = - jax.lax.stop_gradient(jnp.exp(params[1])) * entropy + \
                jnp.exp(params[1]) * jax.lax.stop_gradient(entropy - H_target * jnp.log(self.outDim))

            return lossV + lossP + lossE
        
        return jnp.mean(jax.vmap(get_loss)(result.value, result.logits, self.gamma))
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def PG_step(self, state, params, tau, H_target=0.9):
        loss, grad = jax.value_and_grad(self.PG_loss)(params, tau, H_target)

        updates, state = self.opti.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss

    def ETD_loss(self, params, tau:Tau, H_target:jnp.ndarray):

        assert len(tau.obs)    >= 2*self.n 
        assert len(tau.action) >= 2*self.n-1
        assert len(tau.logits) >= 2*self.n-1
        assert len(tau.reward) >= 2*self.n-1
        assert len(tau.done)   >= 2*self.n-1

        result = self.apply_fn(params[0], self.obs_process(tau.obs))

        ln_mu = jnp.sum(
            jax.nn.log_softmax(tau.logits.at[:2*self.n-1].get()) * 
            jax.nn.one_hot(tau.action.at[:2*self.n-1].get(), self.outDim),
        axis=-1)

        def get_loss(value, logits, Ftrace, gamma):
            gamma = gamma * (1-tau.done.at[:2*self.n-1].get())

            ln_pi = jnp.sum(
                jax.nn.log_softmax(logits.at[:2*self.n-1].get()) * 
                jax.nn.one_hot(tau.action.at[:2*self.n-1].get(), self.outDim),
            axis=-1)

            RHO = sg(
                jnp.minimum(1, jnp.exp(ln_pi - ln_mu))
            )

            if not self.trust_region is None:
                Dkl = jnp.sum(
                    jax.nn.softmax(logits.at[:2*self.n-1].get()) * (
                        jax.nn.log_softmax(logits.at[:2*self.n-1].get()) - 
                        jax.nn.log_softmax(tau.logits.at[:2*self.n-1].get())
                ), axis=-1)

                RHO = RHO * jnp.less(Dkl, self.trust_region)

            delta = RHO * sg(
                tau.reward.at[:2*self.n-1].get() + gamma * 
                value.at[1:2*self.n].get() - value.at[:2*self.n-1].get()
            )

            
            E = vmap(lambda t : jnp.sum(vmap(lambda i :
                
                delta.at[i].get() * jnp.prod(vmap(lambda j : 
                    gamma.at[j].get() * RHO.at[j].get() * jnp.less(j, i) * jnp.greater_equal(j, t) +
                    1 - jnp.less(j, i) * jnp.greater_equal(j, t)
                )(jnp.arange(2*self.n-1)), axis=0) #jnp.arange(t, i)
            
            )(t+jnp.arange(self.n)), axis=0))(jnp.arange(self.n))

            V = sg(value).at[:self.n+1].get().at[:self.n].add(E)

            lossV = (value.at[:self.n].get() - V.at[:self.n].get()) ** 2

            lossP = -ln_pi.at[:self.n].get() * RHO.at[:self.n].get() * sg(
                tau.reward.at[:self.n].get() + gamma.at[:self.n].get() * V.at[1:self.n+1].get() - value.at[:self.n].get()
            )

            entropy = - jnp.sum(
                jax.nn.softmax(logits.at[:self.n].get()) *
                jax.nn.log_softmax(logits.at[:self.n].get()), 
            axis=-1)

            lossE = - jax.lax.stop_gradient(jnp.exp(params[1])) * entropy + \
                jnp.exp(params[1]) * jax.lax.stop_gradient(entropy - H_target * jnp.log(self.outDim))

            if not self.DKL_target is None:
                Dkl = jnp.sum(
                    jax.nn.softmax(logits.at[:self.n].get()) * (
                        jax.nn.log_softmax(logits.at[:self.n].get()) - 
                        jax.nn.log_softmax(tau.logits.at[:self.n].get())
                ), axis=-1)

                lossP = lossP + jnp.exp(params[2]) * sg(self.DKL_target - Dkl) + sg(jnp.exp(params[2])) * Dkl

            if self.use_Ftrace:

                trace_terme = vmap(lambda t : jnp.prod(vmap(lambda i :

                    gamma.at[i].get() * RHO.at[i].get()

                )(t+jnp.arange(self.n)), axis=0))(jnp.arange(self.n))

                lossF = (
                    trace_terme * sg(Ftrace.at[:self.n].get()) + 1 - Ftrace.at[self.n:2*self.n].get()
                ) ** 2

                return (lossV + lossP + lossE) * Ftrace.at[:self.n].get() + lossF
            
            else:
                return lossV + lossP + lossE
        
        return jnp.mean(jax.vmap(get_loss)(result.value, result.logits, result.Ftrace, self.gamma))
    

    @functools.partial(jax.jit, static_argnums=(0,))
    def ETD_step(self, state, params, tau, H_target=0.9):
        loss, grad = jax.value_and_grad(self.ETD_loss)(params, tau, H_target)

        updates, state = self.opti.update(grad, state, params)
        params = optax.apply_updates(params, updates)

        return state, params, loss

class PartialTau:
    def __init__(self, trajectory_n, use_ETD=False):
        self.use_ETD = use_ETD
        self.n = trajectory_n
        self.tau = None
    
    def add_transition_V_TRACE(self, obs, logits, action, reward, done, n_obs):
        if self.tau is None: self.tau = Tau(obs=[obs], action=[], reward=[], done=[], logits=[])
        
        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.action.append(action)
        self.tau.reward.append(reward)
        self.tau.logits.append(logits)

        if len(self.tau.done) == self.n:


            tau = Tau(
                obs   =np.array(self.tau.obs),
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                action=np.array(self.tau.action),
                logits=np.array(self.tau.logits),
            )

            self.tau = Tau(
                obs   =[self.tau.obs[-1]],
                done  =[], 
                reward=[], 
                action=[], 
                logits=[], 
            )
            return tau 
        return None

    def add_transition_ETD(self, obs, logits, action, reward, done, n_obs):
        if self.tau is None: self.tau = Tau(obs=[obs], action=[], reward=[], done=[], logits=[])
        
        self.tau.obs.append(n_obs)
        self.tau.done.append(done)
        self.tau.action.append(action)
        self.tau.reward.append(reward)
        self.tau.logits.append(logits)
        

        if len(self.tau.obs) == 2*self.n:
            tau = Tau(
                #obs   =np.array(self.tau.obs),
                done  =np.array(self.tau.done),
                reward=np.array(self.tau.reward),
                action=np.array(self.tau.action),
                logits=np.array(self.tau.logits),
                obs   =jax.tree_multimap(lambda *args: np.array(args), *self.tau.obs),
            )

            self.tau = Tau(
                obs   =[self.tau.obs[i]    for i in range(self.n, 2*self.n)], 
                done  =[self.tau.done[i]   for i in range(self.n, 2*self.n-1)], 
                reward=[self.tau.reward[i] for i in range(self.n, 2*self.n-1)], 
                action=[self.tau.action[i] for i in range(self.n, 2*self.n-1)], 
                logits=[self.tau.logits[i] for i in range(self.n, 2*self.n-1)], 
            )

            return tau 
        return None


    def add_transition(self, obs, logits, action, reward, done, n_obs):
        if self.use_ETD: return self.add_transition_ETD(obs, logits, action, reward, done, n_obs)
        return self.add_transition_V_TRACE(obs, logits, action, reward, done, n_obs)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from model import *

    opti = optax.chain(
        #optax.clip_by_global_norm(1.0), 
        optax.adam(learning_rate=2e-4)
    )

    n = 5

    actor = V_TRACE(
        MLP_MODEL, inDim=(4,), outDim=2, 
        num_heads=1, trajectory_n=n, 
        gamma=jnp.array([0.99]), 
        opti=opti, E_coef=0.02
    )

    params = actor.init_params(PRNGKey(42))
    state = actor.init_state(params)


    import gym 
    env = gym.make("CartPole-v1")
    obs = env.reset()
    done = False
    r_sum = 0
    step = 0

    import Buffer
    online_buffer = []
    buffer = Buffer.Buffer(200)
    partial_tau = PartialTau(n)

    writer = SummaryWriter()

    while True:
        logits, softmax = jax.tree_map(
            lambda t : np.array(t), actor.get_main_proba(params, obs)
        )

        action = np.random.choice(actor.outDim, p=softmax / np.sum(softmax))

        n_obs, reward, done, _ = env.step(action)
        r_sum += reward

        if done: 
            print(step, r_sum)
            writer.add_scalar("reward", r_sum, step)
            n_obs = env.reset()
            r_sum = 0

        tau = partial_tau.add_transition(obs, logits, action, reward/100.0, done, n_obs)
        if not tau is None: online_buffer.append(tau)

        if len(online_buffer) == 32:
            #buffer.add_tau_list(online_buffer)
            batch = online_buffer #+ buffer.sample_batch(0)[0]

            online_buffer = []
            #if buffer.size < 128: continue

            

            batch = Tau(
                obs   =jnp.array([[b.obs[i]    for b in batch] for i in range(n+1)]), 
                reward=jnp.array([[b.reward[i] for b in batch] for i in range(n)]), 
                done  =jnp.array([[b.done[i]   for b in batch] for i in range(n)]), 
                action=jnp.array([[b.action[i] for b in batch] for i in range(n)]), 
                logits=jnp.array([[b.logits[i] for b in batch] for i in range(n)])
            )

            state, params, loss = actor.V_TRACE_step(state, params, batch)
        
        step += 1
        obs = n_obs

