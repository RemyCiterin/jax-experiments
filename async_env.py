# attention : très lent

from multiprocessing import Pipe, Process
import numpy as np 
import threading
import queue
import time

def run_env(env_fun, inputPipe, outputPipe, m=1):
    outputPipe, y = outputPipe
    x, inputPipe  = inputPipe
    x.close()
    y.close()

    env_list = []

    for _ in range(m):
        time.sleep(0.1)
        env_list.append(env_fun())
    obs = [env.reset() for env in env_list]

    shape, dtype = obs[0].shape, obs[0].dtype
    outputPipe.send((shape, dtype.name))

    outputPipe.send_bytes(
        np.array(obs, dtype=dtype).tobytes()
    )

    while True:
        action, i = inputPipe.recv()
        obs, reward, done, _ = env_list[i].step(action)
        if done: obs = env_list[i].reset()

        obs = np.array(obs, dtype=dtype)
        outputPipe.send_bytes(obs.tobytes())
        outputPipe.send(reward)
        outputPipe.send(done)
        outputPipe.send(i)

class VecEnv(object):
    def __init__(self, env_fun, n, m=1):
        inputPipe  = [Pipe() for _ in range(n)]
        outputPipe = [Pipe() for _ in range(n)]

        for i in range(n):
            process = Process(target=run_env, args=(env_fun, outputPipe[i], inputPipe[i], m))
            process.start()

            inputPipe[i][0].close()
            outputPipe[i][1].close()

        self.inputPipe  = [p[1] for p in inputPipe]
        self.outputPipe = [p[0] for p in outputPipe]

        for i in range(n):
            self.shape, self.dtype = self.inputPipe[i].recv()

        self.init_obs = np.zeros((n, m, *self.shape), dtype=self.dtype)
        for i in range(n): self.init_obs[i] = np.frombuffer(
                self.inputPipe[i].recv_bytes(), dtype=self.dtype).reshape((m, *self.shape))

        self.n = n
        self.m = m

        self.queue = queue.Queue()

        threading.Thread(target=self.work).start()

    def get_init_obs(self):
        for i in range(self.n):
            for j in range(self.m):
                self.queue.put((self.init_obs[i, j], 0.0, False, (i, j)))
        return self.init_obs

    def set_action(self, action, ident):
        self.outputPipe[ident[0]].send((action, ident[1]))

    def work(self):
        while True:
            for i in range(self.n):
                obs = np.frombuffer(
                    self.inputPipe[i].recv_bytes(), dtype=self.dtype).reshape(self.shape)

                reward = self.inputPipe[i].recv()
                done = self.inputPipe[i].recv()
                j = self.inputPipe[i].recv()

                self.queue.put((obs, reward, done, (i, j)))

