import grpc


import SEED.bidirectional_pb2 as bidirectional_pb2
import SEED.bidirectional_pb2_grpc as bidirectional_pb2_grpc

import vizdoomgym
import gym 
import multiprocessing as mp 

from torch.utils.tensorboard import SummaryWriter
from Wrapper import Doom, Transpose, DoneIfReward, FromUInt8, Frameskip, Wrapper, SkipFrames, Buffer, ToContinuous, MeanRGB
import numpy as np
import threading
import time

def main(result_queue, NUM_ENVS=10, addr='localhost:51005', verbose=False):
    print("process start with verbose={}".format(verbose))
    import queue

    channel = grpc.insecure_channel(addr)
    stub = bidirectional_pb2_grpc.ModelCallStub(channel)

    import gym

    #env_list = [Buffer(SkipFrames(Transpose(Doom(gym.make("VizdoomBasic-v0"))), number=10), number=10) for _ in range(NUM_ENVS)]
    env_list = [Transpose(gym.make("MinAtar/Breakout-v0")) for _ in range(NUM_ENVS)]
    #env_list = [Frameskip(Wrapper(gym.make("PongNoFrameskip-v0"))) for _ in range(NUM_ENVS)]
    #env_list = [FromUInt8(gym.make("Pong-ram-v0")) for _ in range(NUM_ENVS)]
    #env_list = [gym.make("LunarLander-v2") for _ in range(NUM_ENVS)]

    if verbose:
        writer = SummaryWriter()
    
    r_sum = [0.0 for _ in range(NUM_ENVS)]
    steps = 0

    Q = queue.Queue(NUM_ENVS+1)

    for i in range(NUM_ENVS):
        Q.put((i, env_list[i].reset(), verbose*1, False))
    
    def generate_obs():
        while True:
            env_id, obs, reward, done = Q.get()

            yield bidirectional_pb2.Observation(
                data = obs.tobytes(), dimension = obs.shape, dtype = obs.dtype.name, 
                reward = reward, done = done, env_id = env_id
            )

    for action in stub.MakeAction(generate_obs()):

        obs, reward, done, _ = env_list[action.env_id].step(action.action)
        r_sum[action.env_id] += reward 
        steps += 1

        if done:
            obs = env_list[action.env_id].reset()
            if verbose: writer.add_scalar("reward", r_sum[action.env_id], steps)
            else: result_queue.put(("reward", r_sum[action.env_id]))
            print(r_sum[action.env_id], steps)
            r_sum[action.env_id] = 0
        
        Q.put((action.env_id, obs, reward, done))

if __name__ == "__main__":
    result_queue = mp.Queue(128)
    verbose = True

    process = [mp.Process(target=main, args=(result_queue, 10, 'localhost:'+port, i == 0 and verbose))
        for port in ['51006'] for i in range(10)
    ]
    [p.start() for p in process]

    if not verbose:
        writer = SummaryWriter()
        start_time= time.time()

        while True:
            name, val = result_queue.get()
            writer.add_scalar(name, val, time.time()-start_time)
    else:
        while True:
            result_queue.get()

    [p.join() for p in process]