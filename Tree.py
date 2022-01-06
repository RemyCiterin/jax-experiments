import numpy as np 

class SumTree(object):
    def __init__(self, size, array=None):
        self.size = size 

        self.array = np.zeros(2*size)

        if not array is None: 
            for i, a in enumerate(array):
                self[i] = a 
    
    def __setitem__(self, i, a):
        self.array[i+self.size] = a 

        index = (i+self.size) // 2

        while index > 0:
            self.array[index] = self.array[2*index] + (
                self.array[2*index+1] if 2*index+1 < self.size*2 else 0
            )

            index = index // 2
    
    def __getitem__(self, i):
        return self.array[i+self.size]
    
    def sample(self):
        x = np.random.uniform() * self.array[1]

        index = 1 

        while index < self.size:
            if x > self.array[index*2]:
                x -= self.array[index*2]
                index = 2*index+1
            else:
                index = 2*index

        return index - self.size
    


class MinTree(object):
    
    infty = 1e9

    def __init__(self, size, array=None):
        self.size = size 

        self.array = np.zeros(2*size) + MinTree.infty

        if not array is None: 
            for i, a in enumerate(array):
                self[i] = a
    
    def __setitem__(self, i, a):
        self.array[i+self.size] = a 

        index = (i+self.size) // 2

        while index > 0:
            self.array[index] = (
                min(self.array[2*index], self.array[2*index+1]) if 2*index+1 < self.size*2 else self.array[2*index]
            )

            index = index // 2
    
    def __getitem__(self, i):
        return self.array[i+self.size]
