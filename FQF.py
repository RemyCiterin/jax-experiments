from jax.random import PRNGKey, split 
import jax.numpy as jnp
from jax import vmap
import optax 
import jax 

import haiku as hk 
import numpy as np 

from collections import namedtuple
from typing import Any, Callable, NamedTuple
import functools

sg = jax.lax.stop_gradient


Tau = namedtuple('Tau', ["obs", "reward", "gamma", "action", "n_obs"])

class PROPOSAL_MODEL(hk.Module):
    def __init__(self, n_atoms):
        super().__init__()

        self.n_atoms = n_atoms 
    
    def __call__(self, repr):
        assert len(repr.shape) == 1

        return hk.Sequential([
            hk.Linear(self.n_atoms), 
            jax.nn.softmax
        ])(repr)

class VALUE_MODEL(hk.Module):
    def __init__(self):
        super().__init__()
    
    def __call__(self, repr, tau):
        assert len(repr.shape) == 1
        assert len(tau.shape)  == 0

        hDim = repr.shape[0]

        phi = jax.vmap(lambda i : i * jnp.pi * tau)(jnp.arange(hDim))
        phi = jax.nn.relu(hk.Linear(hDim)(jnp.cos(phi)))

        return jnp.squeeze(hk.Sequential([
                hk.Linear(hDim), jax.nn.relu, 
                hk.Linear(1)
            ])(phi * repr), 
        axis=-1)

class PARAMS(NamedTuple):
    representation:Any
    proposal:Any
    value:Any

class FQF:
    def __init__(self, core:Callable[[int, int], hk.Module], inDim, outDim:int, 
            n_atoms:int=32, opti=optax.adam(2e-4), hDim=256):

        self._init_repr, self._apply_repr = hk.without_apply_rng(
            hk.transform(lambda x : core(hDim, num_heads=1)(x[None]).logits[0, 0])
        )

        self._init_proposal, self._apply_proposal = hk.without_apply_rng(
            hk.transform(lambda x : PROPOSAL_MODEL(n_atoms)(x))
        )

        self._init_value, self._apply_value = hk.without_apply_rng(
            hk.transform(lambda x, t : VALUE_MODEL()(x, t))
        )

        self.opti = opti

        self.n_atoms = n_atoms 
        self.outDim = outDim 
        self.inDim = inDim
        self.hDim = hDim 
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def init_params(self, key):
        k1, k2 = split(key)

        repr_params = self._init_repr(k1, jnp.zeros(self.inDim)[None])

        k3, k4 = split(k2)

        proposal_params = jax.vmap(
            lambda k : self._init_proposal(k, jnp.zeros((self.hDim,)))
        )(split(k3, self.outDim))

        value_params = jax.vmap(
            lambda k : self._init_value(k, jnp.zeros((self.hDim,)), jnp.zeros([]))
        )(split(k4, self.outDim))

        return PARAMS(
            repr_params, proposal_params, value_params
        )
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs):
        repr = self._apply_repr(params.representation, obs)

        def aux(hidden):
            # \sum soft[i] = 1
            soft = jax.vmap(
                lambda p : self._apply_proposal(p, hidden)
            )(params.proposal)

            eval_tau = jax.jit(lambda i : jnp.sum(soft * jnp.less(jnp.arange(self.n_atoms), i), axis=-1))

            tau = jax.vmap(eval_tau)(jnp.arange(self.n_atoms+1)).T
            tau_ = (tau.T.at[1:].get() + tau.T.at[:-1].get()).T / 2.0

            Z = jax.vmap(
                lambda p,t_vec : jax.vmap(lambda t :self._apply_value(p, hidden, t))(t_vec)
            )(params.value, tau_)

            Q = jnp.sum(
                (tau.T.at[1:].get() - tau.T.at[:-1].get()).T * Z
            , axis=-1)

            return jnp.argmax(Q)

        return jax.vmap(aux)(repr)

from model import * 
fqf = FQF(MLP_MODEL, (4,), 2, 32, optax.adam(2e-4), 256)

params = fqf.init_params(PRNGKey(42))
print(fqf.get_action(params, jnp.zeros((5, 4))).shape)



