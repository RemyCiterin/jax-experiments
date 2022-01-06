import jax.numpy as jnp
from jax import vmap
import jax 

import haiku as hk 
from typing import NamedTuple
import functools

sg = jax.lax.stop_gradient

class MODEL_OUTPUT(NamedTuple):
    logits:jnp.ndarray
    Ftrace:jnp.ndarray
    value:jnp.ndarray

class MLP_MODEL(hk.Module):
    def __init__(self, outDim, num_heads=3):
        super().__init__()

        self.outDim = outDim 
        self.N = num_heads

    
    def __call__(self, obs):
        h = hk.Sequential([
            hk.Linear(256), jax.nn.relu, 
            hk.Linear(256), jax.nn.relu
        ])(obs)

        return jax.vmap(lambda _ :MODEL_OUTPUT(
            Ftrace=jax.nn.relu(jnp.squeeze(hk.Linear(1)(sg(h)), axis=-1)),
            value=jnp.squeeze(hk.Linear(1)(h), axis=-1),
            logits=hk.Linear(self.outDim)(h)
        ))(jnp.arange(self.N))

class LittleConvModel(hk.Module):
    def __init__(self, outDim, num_heads=3):
        super().__init__()

        self.outDim = outDim 
        self.N = num_heads

    
    def __call__(self, obs):
        convModel = hk.Sequential([
            hk.Conv2D(32, 3, 1, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(32, 3, 1, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(32, 3, 1, padding='Valid'), jax.nn.relu
        ])

        h = vmap(vmap(lambda o : 
            jnp.reshape(convModel(jnp.transpose(o, axes=(1, 2, 0))), (-1,))
        ))(obs)
        
        return MLP_MODEL(self.outDim, self.N)(h)

class ConvModel(hk.Module):
    def __init__(self, outDim, num_heads=3):
        super().__init__()

        self.outDim = outDim
        self.N = num_heads

    
    def __call__(self, obs):
        convModel = hk.Sequential([
            hk.Conv2D(32, 8, 4, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(32, 4, 2, padding='Valid'), jax.nn.relu, 
            hk.Conv2D(32, 3, 1, padding='Valid'), jax.nn.relu
        ])

        h = vmap(vmap(lambda o : 
            jnp.reshape(convModel(jnp.transpose(o, axes=(1, 2, 0))), (-1,))
        ))(obs)
        
        return MLP_MODEL(self.outDim, self.N)(h)