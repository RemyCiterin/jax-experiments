import os 
os.environ["XLA_GPU_STRICT_CONV_ALGORITHM_PICKER"] = "false"
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.30'
import jax
import jax.numpy as jnp

from V_TRACE import *
from model import *

import gym
import numpy as np
from functools import partial

actor = V_TRACE(
    ConvModel, (4, 84, 84), 4,
    1, 1, jnp.array([0.99])
)

actor.obs_process = jax.jit(lambda obs : obs / 255)

import pickle

file = open("MODEL/Breakout-v5_280000.pickle", "rb")
params = pickle.load(file)
file.close()

import time

@jax.jit
def get_action(rng, params, obs):
    rng1, rng2 = jax.random.split(rng, num=2)
    logits, softmax = jax.tree_map(lambda t : t[0, 0], actor.get_main_proba(params, obs[None, None]))
    #softmax = jnp.power(softmax, 5) / jnp.sum(jnp.power(softmax, 5))
    #action = jax.random.choice(rng1, a=actor.outDim, p=softmax)
    action = jnp.argmax(softmax)
    return action, logits, rng2



from wrapper_deepmind import wrap_deepmind, make_atari


def doom():
	import cv2
	import envpool
	import vizdoomgym
	rng = jax.random.PRNGKey(42)
	env = envpool.make_gym("DeadlyCorridor-v1", num_envs=1, use_combined_action=True)
	out = cv2.VideoWriter('output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (84, 84))

	obs = env.reset()
	for _ in range(1000):
		action, _, rng = get_action(rng, params, obs[0])
		obs, reward, done, _ = env.step(np.array(action)[None])
		print(np.transpose(obs[0, :3], (1, 2, 0)).shape)
		out.write(np.transpose(obs[0, :3], (1, 2, 0))[::, ::, ::-1])
	out.release()


#doom()
#assert False



rng = jax.random.PRNGKey(42)
env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, episode_life=True, frame_stack=True, scale=False, grayscale=True)
obs = env.reset()


while True:
	obs = obs.__array__()
	action, _, rng = get_action(rng, params, np.transpose(obs, (2, 0, 1)))
	obs, reward, done, _ = env.step(np.array(action))
	if done: obs = env.reset()

	time.sleep(0.01)
	#env.render()