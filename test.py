import os 

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'

import jax 
import haiku as hk
import optax 
import numpy as np 
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.random import PRNGKey, split

import functools

from typing import NamedTuple, Tuple, Callable, Any

import jax.random

class PongState(NamedTuple):
    leftPos   : jnp.ndarray
    rightPos  : jnp.ndarray
    ballPos   : jnp.ndarray
    ballSpeed : jnp.ndarray

"""
Pong environement  : all pos is between 0 and 1
"""

def is_in_rectangle(a, b, c, d, x, y):
    return jnp.less(a, x) * jnp.less_equal(x, c) * jnp.less(b, y) * jnp.less_equal(y, d)

def test_colision(rec1, rec2):
    (a, b, c, d) = rec1
    (x, y, z, t) = rec2 

    test = is_in_rectangle(a, b, c, d, x, y) + \
        is_in_rectangle(a, b, c, d, x, t) + \
        is_in_rectangle(a, b, c, d, z, t) + \
        is_in_rectangle(a, b, c, d, z, y) + \
        is_in_rectangle(x, y, z, t, a, b) + \
        is_in_rectangle(x, y, z, t, a, d) + \
        is_in_rectangle(x, y, z, t, c, b) + \
        is_in_rectangle(x, y, z, t, c, d)
    
    return jnp.not_equal(test, 0).astype(float)


class JaxPong:
    def __init__(self, speedL, speedR, max_ball_speed, delta_t, paddleW, paddleH, ballD, lossL):
        self.max_ball_speed = max_ball_speed
        self.delta_t = delta_t
        self.paddleW = paddleW
        self.paddleH = paddleH
        self.speedL = speedL
        self.speedR = speedR
        self.ballD = ballD
        self.lossL = lossL
        
    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, rng) -> Tuple[Any, PongState]:
        rng1, rng2, rng3, rng4, rng5 = split(rng, 5)
        
        leftPos   = jax.random.uniform(rng1)
        rightPos  = jax.random.uniform(rng2)
        ballSpeed = jax.random.uniform(rng3, shape=(2,))
        ballPos   = jnp.zeros((2,)) + 0.5

        ballSpeed = jnp.sign(ballSpeed * 2 - 1) * self.max_ball_speed * jax.random.uniform(rng4)

        return rng5, PongState(leftPos, rightPos, ballPos, ballSpeed)
    
    @functools.partial(jax.jit, static_argnums=(0, 2, 3))
    def get_obs(self, state:PongState, n, m):
        
        def get_pixel(i, j):
            p1 = is_in_rectangle(
                self.lossL - self.paddleW / 2.0, state.leftPos - self.paddleH / 2.0, 
                self.lossL + self.paddleW / 2.0, state.leftPos + self.paddleH / 2.0, 
                i / n, j / m
            )

            p2 = is_in_rectangle(
                1 - self.lossL - self.paddleW / 2.0, state.rightPos - self.paddleH / 2.0, 
                1 - self.lossL + self.paddleW / 2.0, state.rightPos + self.paddleH / 2.0, 
                i / n, j / m
            )

            p3 = is_in_rectangle(
                state.ballPos[0] - self.ballD / 2.0, state.ballPos[1] - self.ballD / 2.0, 
                state.ballPos[0] + self.ballD / 2.0, state.ballPos[1] + self.ballD / 2.0, 
                i / n, j / m
            )

            return jnp.array((p1, p2, p3))

        return jnp.transpose(jax.vmap(lambda i : jax.vmap(lambda j : get_pixel(j, i)
        )(jnp.arange(n)))(jnp.arange(m)), (2, 0, 1))


    
    @functools.partial(jax.jit, static_argnums=0)
    def _step(self, state:PongState, action):
        # action is a real between -1 and 1

        rightPos = jnp.clip(
            state.rightPos + self.delta_t * jnp.clip(action, -1, 1) * self.speedR, 0, 1
        )

        leftPos = jnp.clip(
            state.leftPos + self.speedL * self.delta_t * jnp.sign(state.ballPos.at[1].get() - state.leftPos), 0, 1
        )

        speedR = (rightPos - state.rightPos) / self.delta_t 
        speedL = (leftPos - state.leftPos) / self.delta_t 

        ballPos = state.ballPos + state.ballSpeed * self.delta_t

        reward = jnp.less(ballPos.at[0].get(), 0.0).astype(float) - jnp.greater(ballPos.at[0].get(), 1.0).astype(float)

        invSpeedY = jnp.less(ballPos.at[1].get(), 0.0).astype(float) + jnp.less(1.0, ballPos.at[1].get()).astype(float)

        ballSpeedY = state.ballSpeed.at[1].get() * (1 - 2 * invSpeedY)

        invSpeedXL = test_colision(
            (
                ballPos.at[0].get() - self.ballD / 2.0, ballPos.at[1].get() - self.ballD / 2.0, 
                ballPos.at[0].get() + self.ballD / 2.0, ballPos.at[1].get() + self.ballD / 2.0
            ), (
                self.lossL - self.paddleW / 2.0, state.leftPos - self.paddleH / 2.0,
                self.lossL + self.paddleW / 2.0, state.leftPos + self.paddleH / 2.0
            )
        )
        invSpeedXR = test_colision(
            (
                ballPos.at[0].get() - self.ballD / 2.0, ballPos.at[1].get() - self.ballD / 2.0, 
                ballPos.at[0].get() + self.ballD / 2.0, ballPos.at[1].get() + self.ballD / 2.0
            ), (
                1 - self.lossL - self.paddleW / 2.0, state.rightPos - self.paddleH / 2.0, 
                1 - self.lossL + self.paddleW / 2.0, state.rightPos + self.paddleH / 2.0
            )
        )

        invSpeedX = invSpeedXL + invSpeedXR

        ballSpeedX = state.ballSpeed.at[0].get() * (1 - 2 * invSpeedX)

        ballSpeedY = ballSpeedY + (1 - invSpeedY) * 0.5 * speedL * invSpeedXL
        ballSpeedY = ballSpeedY + (1 - invSpeedY) * 0.5 * speedR * invSpeedXR

        ballSpeed = jnp.array((ballSpeedX, ballSpeedY))

        ballSpeed = jnp.clip(ballSpeed, -self.max_ball_speed, self.max_ball_speed)

        return PongState(leftPos, rightPos, ballPos, ballSpeed), jnp.squeeze(reward), jnp.squeeze(jnp.not_equal(reward, 0))

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state, action, rng):
        rng, state_if_done = self.reset(rng)
        state, reward, done = self._step(state, action)
        state = jax.tree_map(lambda s, s_ : (1 - done) * s + done * s_, state, state_if_done)
        return jax.tree_map(jax.lax.stop_gradient, (state, reward, done, rng))
@jax.jit
def frameStack(last_obs, obs):
    return last_obs.at[3:].set(last_obs[:-3]).at[:3].set(obs)

size = 256
n, m = 50, 50

pong = JaxPong(speedL=1.0, speedR=1.25, max_ball_speed=1.5, delta_t=0.05, paddleW=0.05, paddleH=0.2, ballD=0.05, lossL=0.1)


rng, state = pong.reset(PRNGKey(42))
"""
for _ in range(20):
    state, reward, done, rng = pong.step(state, 0.1, rng)
    img = pong.get_obs(state, n, m)
    plt.imshow(np.transpose(img, (1, 2, 0)).astype(float))
    plt.show()
    print(reward)
"""

from V_TRACE import *
from model import *

N = 1
opti = optax.adam(2e-4)

actor = V_TRACE(
    MLP_MODEL, inDim=(6,), outDim=2,
    num_heads=1, trajectory_n=N,
    gamma=jnp.array([0.99]),
    opti=opti, E_coef=0.9
)
"""
actor = V_TRACE(
    ConvModel, inDim=(12, n, m), outDim=2,
    num_heads=1, trajectory_n=N,
    gamma=jnp.array([0.99]),
    opti=opti, E_coef=0.8
)
"""


@jax.jit
def state_to_obs(state:PongState):
    return jnp.array((
        state.leftPos,
        state.rightPos,
        state.ballPos.at[0].get(), 
        state.ballPos.at[1].get(), 
        state.ballSpeed.at[0].get(), 
        state.ballSpeed.at[1].get(),
    ))

params = actor.init_params(PRNGKey(42))
opti_state = actor.init_state(params)

partial_tau = PartialTau(N)

@functools.partial(jax.jit, static_argnums=0)
def multi_steps(n, state, action, rng):
    def aux(carry, _):
        state, reward, done, rng = carry
        state, rew, d, rng = pong.step(state, action, rng)
        carry = (state, reward + rew*(1-done), jnp.logical_or(d, done), rng)
        return carry, carry
    return jax.lax.scan(aux, (state, jnp.array(0), jnp.array(False, dtype=bool), rng), None, n)[0]

import time

def train(total_time):
    rng, state = jax.vmap(lambda i : pong.reset(PRNGKey(i)))(jnp.arange(size))
    #obs = jax.vmap(lambda s : pong.get_obs(s, n, m))(state).astype(float)
    #obs = jax.vmap(functools.partial(frameStack, jnp.zeros((12, n, m))))(obs)
    obs = jax.vmap(state_to_obs)(state)
    
    global params
    global opti_state
    global reward_curve
    steps = 0
    current_r_sum = jnp.zeros((size,))
    last_r_sum    = jnp.zeros((size,))
    episode_len   = jnp.zeros((size,))
    reward_curve = []

    start_time = time.time()
    while time.time() - start_time < total_time:
        logits, softmax = jax.tree_map(
            lambda t : np.array(t)[0], actor.get_main_proba(params, obs[None, ...])
        )

        actions = np.array([np.random.choice(actor.outDim, p=s / np.sum(s)) for s in softmax], dtype=int)

        #state, reward, done, rng = jax.vmap(functools.partial(multi_steps, 4))(state, actions * 2 - 1, rng)
        state, reward, done, rng = jax.vmap(pong.step)(state, actions * 2 - 1, rng)

        #n_obs = jax.vmap(lambda s : pong.get_obs(s, n, m))(state).astype(float)
        #n_obs = jax.vmap(lambda o, l_o, d : frameStack(l_o*(1-d), o))(n_obs, obs, done)
        n_obs = jax.vmap(state_to_obs)(state)

        current_r_sum = reward + current_r_sum
        last_r_sum    = (1-done) * last_r_sum + done * current_r_sum
        episode_len   = (1 + episode_len) * (1-done)
        current_r_sum = current_r_sum * (1-done)


        tau = partial_tau.add_transition(obs, logits, actions, reward, done, n_obs)
        steps += size

        if not tau is None:
            tau = Tau(
                obs = jnp.array(tau.obs), 
                done = jnp.array(tau.done), 
                reward = jnp.array(tau.reward), 
                action = jnp.array(tau.action), 
                logits = jnp.array(tau.logits)
            )
            opti_state, params, loss = actor.V_TRACE_step(opti_state, params, tau)
            print(end="\r{}   {}   {}   {}   {}   {}   {}   ".format(
                str(loss)[:7], steps, int(time.time()-start_time), 
                int(np.mean(episode_len)), str(np.mean(last_r_sum))[:6], 
                str(np.exp(params[1]))[:6], str(np.sum(-softmax * np.log(softmax))/size)[:5]
            ))

            reward_curve.append(np.mean(last_r_sum))
        obs = n_obs
    print()

train(360)

plt.plot(reward_curve)
plt.show()