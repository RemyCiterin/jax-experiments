import threading
import grpc
from concurrent import futures
import time

import numpy as np

# import the generated classes
import bidirectional_pb2
import bidirectional_pb2_grpc

from ETD import *
from model import *
N_steps = 1

opti = optax.chain(
    optax.adam(learning_rate=2e-4)
)


agent = ETD(LittleConvModel, 
    inDim=(4, 10, 10), outDim=6,
    num_heads=3, trajectory_n=N_steps,
    gamma=jnp.array([0.99, 0.985, 0.98]),
    #trust_region=np.log(6) / 5,
    opti=opti, E_coef=0.03,
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

class ModelCall(bidirectional_pb2_grpc.ModelCallServicer):
    def __init__(self, learner_queue, agent):
        super().__init__()

        self.learner_queue = learner_queue
        self.agent = agent

        self.num_worker = 0

        self.lock = threading.Lock()

    def MakeAction(self, request_iterator, context):

        self.lock.acquire()
        worker_id = self.num_worker
        self.num_worker += 1
        self.lock.release()

        partial_tau = None#PartialTau(N_steps)
        last_state  = None

        for request in request_iterator:

            env_id = np.frombuffer(request.id_data, dtype=request.id_dtype).reshape(request.id_dim)
            done   = np.frombuffer(request.done_data, dtype=request.done_dtype).reshape(request.done_dim)
            reward = np.frombuffer(request.reward_data, dtype=request.reward_dtype).reshape(request.reward_dim) 
            obs    = np.frombuffer(request.obs_data, dtype=request.obs_dtype).reshape(request.obs_dim).astype(float)

            logits, softmax = jax.tree_map(
                lambda t : np.array(t)[0],
                self.agent.get_main_proba(params, obs[None, ...])
            )

            action = np.array([
                np.random.choice(self.agent.outDim, p=s / np.sum(s)) for s in softmax
            ])

            yield bidirectional_pb2.Action(
                action_data = action.tobytes(), action_dim = action.shape, action_dtype = action.dtype.name, 
                id_data = env_id.tobytes(), id_dim = env_id.shape, id_dtype = env_id.dtype.name
            )

            if partial_tau is None:
                partial_tau = [PartialTau(N_steps) for i in range(len(obs))]

            if not last_state is None:
                for i in range(len(obs)):
                    tau = partial_tau[i].add_transition(
                        obs=last_state[0][i], logits=last_state[1][i], action=last_state[2][i],
                        reward=reward[i], done=done[i], n_obs=obs[i]
                    )

                    if not tau is None: self.learner_queue.put(tau)
            
            last_state = (obs, logits, action)

# create a gRPC server

import queue

learner_queue = queue.Queue(10)

def make_actor(learner_queue, agent, addr='localhost:51005'):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=12))
    bidirectional_pb2_grpc.add_ModelCallServicer_to_server(
        ModelCall(learner_queue, agent), server
    )

    print('Starting server. Listening on port '+addr)
    server.add_insecure_port(addr)
    server.start()

    return server

server_list = [
    make_actor(learner_queue=learner_queue, agent=agent, addr='localhost:'+port)
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
        while len(batch) < 64: batch.append(learner_queue.get())
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
