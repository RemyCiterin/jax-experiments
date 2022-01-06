import Tree 

import numpy as np 

class Buffer(object):
    def __init__(self, size):
        self.capacity = size 
        self.index = 0 
        self.size = 0 

        self.tau = [None for _ in range(size)]
        self.sumTree = Tree.SumTree(size)
        self.minTree = Tree.MinTree(size)
        self.prioMax = 1

    def add_tau(self, tau, prio=None):
        prio = self.prioMax if prio is None else prio
        self.prioMax = max(prio, self.prioMax)

        self.sumTree[self.index] = prio
        self.minTree[self.index] = prio

        t = self.tau[self.index]
        del t

        self.tau[self.index] = tau 

        self.index = (self.index+1) % self.capacity
        self.size = min(self.capacity, self.size+1)
    
    def add_tau_list(self, tau, prio=None):
        for i in range(len(tau)):

            self.add_tau(tau[i], None if prio is None else prio[i])
        
    
    def sample_batch(self, batch_size):
        index = [self.sumTree.sample() for _ in range(batch_size)]

        return [self.tau[i] for i in index], index, [self.sumTree[i] for i in index], self.minTree.array[1]

    def update(self, index, new_prio):
        for i, p in zip(index, new_prio): 
            self.sumTree[i] = p
            self.minTree[i] = p

            self.prioMax = max(self.prioMax, p)


if __name__ == "__main__":
    print("buffer test : ")
    buffer = Buffer(10)

    for i in range(15):
        buffer.add_tau(i, prio=i)
    
    print(buffer.tau)

    print(buffer.sample_batch(10)[0])
    