import threading
import grpc
from concurrent import futures
import time

import numpy as np
import queue


import SEED.bidirectional_pb2 as bidirectional_pb2
import SEED.bidirectional_pb2_grpc as bidirectional_pb2_grpc

from ETD import *
from model import *
N_steps = 5

opti = optax.chain(
    optax.adam(learning_rate=2e-4)
)

Actor_Batch = 32




agent = ETD(LittleConvModel, 
    inDim=(4, 10, 10), outDim=6,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(6) / 5,
    opti=opti, E_coef=0.02,
    use_Ftrace=False,
)

"""
agent = ETD(ConvModel, 
    inDim=(10, 64, 48), outDim=3,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(3)/5,
    opti=opti, E_coef=0.05,
    use_Ftrace=False,
)
"""

params = agent.init_params(PRNGKey(42))
state = agent.init_state(params)

def complete_trajectory(learner_queue):
    partial_tau = {}
    last_state  = {}

    def add_transition(worker_id, env_id, n_obs, reward, done, logits, action):

        ident = (worker_id, env_id).__hash__()

        if not ident in partial_tau:
            partial_tau[ident] = PartialTau(N_steps)

        if ident in last_state:
            tau = partial_tau[ident].add_transition(
                obs=last_state[ident][0], action=last_state[ident][1], 
                logits=last_state[ident][2], reward=reward, done=done, 
                n_obs=n_obs
            )

            if not tau is None:
                learner_queue.put(tau)

        last_state[ident] = (n_obs, action, logits)
    
    return add_transition

class ModelCall(bidirectional_pb2_grpc.ModelCallServicer):
    def __init__(self, learner_queue, agent, ident):
        super().__init__()

        self.agent = agent
        self.ident = ident

        self.num_worker = 0

        self.lock = threading.Lock()

        self.buffer = []

        self.add_transition = complete_trajectory(learner_queue)
        self.action_queue = []

    def MakeAction(self, request_iterator, context):

        self.lock.acquire()
        worker_id = self.num_worker
        self.action_queue.append(queue.Queue(128))
        self.num_worker += 1
        self.lock.release()

        def apply_request():
        
            for request in request_iterator:
                obs = np.frombuffer(request.data, dtype=request.dtype).reshape(request.dimension).astype(float)

                self.lock.acquire()
                self.buffer.append((
                    worker_id, request.env_id, obs, request.reward, request.done
                ))

                if len(self.buffer) == Actor_Batch:
                    obs = np.array([b[2] for b in self.buffer])

                    logits, softmax = jax.tree_map(
                        lambda t : np.array(t)[0],
                        self.agent.get_main_proba(params, obs[None, ...])
                    )

                    action = [
                        np.random.choice(agent.outDim, p=s / np.sum(s)) for s in softmax
                    ]

                    for i in range(len(self.buffer)):
                        self.add_transition(
                            self.buffer[i][0], self.buffer[i][1], self.buffer[i][2],
                            self.buffer[i][3], self.buffer[i][4], logits[i], action[i])

                    for i in range(len(self.buffer)):
                        self.action_queue[self.buffer[i][0]].put((
                            action[i], self.buffer[i][1]
                        ))
                        
                
                    self.buffer = []
                
                self.lock.release()
            
        threading.Thread(target=apply_request).start()

        while True:
            action, env_id = self.action_queue[worker_id].get()
            yield bidirectional_pb2.Action(
                action=action, env_id=env_id
            )

# create a gRPC server

import queue

transition_queue = queue.Queue(128)
learner_queue = queue.Queue(128)

def make_actor(learner_queue, agent, ident, addr='localhost:51005'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=12))
    bidirectional_pb2_grpc.add_ModelCallServicer_to_server(
        ModelCall(learner_queue, agent, ident), server
    )

    print('Starting server. Listening on port '+addr)
    server.add_insecure_port(addr)
    server.start()

    return server

server_list = [
    make_actor(learner_queue=learner_queue, agent=agent, ident=port, addr='localhost:'+port)
    for port in ['51006']
]

try:
    import os
    import time
    import Buffer 
    start_time = time.time()
    #buffer = Buffer.Buffer(200)
    step = 0

    counter = 0
    while True:
        batch = []
        counter -= time.time()
        while len(batch) < 32: batch.append(learner_queue.get())
        counter += time.time()

        #buffer.add_tau_list(batch)
        #batch = batch + buffer.sample_batch(24)[0]
        #if buffer.size < 128: continue

        batch = Tau(
            obs   =jnp.array([[b.obs[i]    for b in batch] for i in range(2*N_steps)]), 
            reward=jnp.array([[b.reward[i] for b in batch] for i in range(2*N_steps-1)]), 
            done  =jnp.array([[b.done[i]   for b in batch] for i in range(2*N_steps-1)]), 
            action=jnp.array([[b.action[i] for b in batch] for i in range(2*N_steps-1)]), 
            logits=jnp.array([[b.logits[i] for b in batch] for i in range(2*N_steps-1)])
        )

        state, params, loss = agent.ETD_step(state, params, batch)

        step += 1
        if step % 10 == 0:
            tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
            #writer.add_scalar("memory", used_m / tot_m, step)
        
            print(end="\rsteps = {}   loss = {}     {}     {}   ".format(
                step, np.mean(loss), counter / (time.time()-start_time), used_m / tot_m
            ))
except KeyboardInterrupt:
    [s.stop(0) for s in server_list]
