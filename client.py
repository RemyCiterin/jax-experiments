import grpc

# import the generated classes
import bidirectional_pb2
import bidirectional_pb2_grpc

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

    event = threading.Event()
    action_list = [None for _ in range(NUM_ENVS)]

    if verbose:
        writer = SummaryWriter()


    def work(queue : queue.Queue):
        def generate_obs():
            while True:
                env_id, obs, reward, done = queue.get()

                yield bidirectional_pb2.Observation(
                    obs_data = obs.tobytes(), obs_dim = obs.shape, obs_dtype = obs.dtype.name, 
                    
                    id_data = env_id.tobytes(), id_dim = env_id.shape, id_dtype = env_id.dtype.name, 

                    reward_data = reward.tobytes(), reward_dim = reward.shape, reward_dtype = reward.dtype.name, 

                    done_data = done.tobytes(), done_dim = done.shape, done_dtype = done.dtype.name, 
                )

        for action in stub.MakeAction(generate_obs()):

            action_batch = np.frombuffer(action.action_data, dtype=action.action_dtype).reshape(action.action_dim)
            env_id_batch = np.frombuffer(action.id_data    , dtype=action.id_dtype    ).reshape(action.id_dim    )

            for env_id, action in zip(env_id_batch, action_batch):
                action_list[env_id] = action
            event.set()

        

    queue = queue.Queue(NUM_ENVS+1)
    worker = threading.Thread(target=work, args=(queue,))
    worker.start()

    queue.put((
        np.arange(NUM_ENVS), np.array([env.reset() for env in env_list]), 
        np.ones([NUM_ENVS]) if verbose else np.zeros([NUM_ENVS]), np.ones([NUM_ENVS])
    ))

    r_sum = [0.0 for _ in range(NUM_ENVS)]
    steps = 0

    while True:
        event.wait()
        obs_batch = []
        done_batch = []
        reward_batch = []
        env_id_batch = []
        for i in range(NUM_ENVS):

            obs, reward, done, _ = env_list[i].step(action_list[i])
            if done: obs = env_list[i].reset()
            action_list[i] = None
            steps += 1

            obs_batch.append(obs)
            done_batch.append(done)
            reward_batch.append(reward)
            env_id_batch.append(i)

            r_sum[i] += reward
            if done:
                if verbose: writer.add_scalar("reward", r_sum[i], steps)
                else: result_queue.put(("reward", r_sum[i]))
                print(r_sum[i], steps)
                r_sum[i] = 0
        
        event.clear()
        queue.put((
            np.array(env_id_batch), np.array(obs_batch),
            np.array(reward_batch), np.array(done_batch)
        ))

if __name__ == "__main__":
    result_queue = mp.Queue(128)
    verbose = True

    process = [mp.Process(target=main, args=(result_queue, 16, 'localhost:'+port, i == 0 and verbose))
        for port in ['51006'] for i in range(4)
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