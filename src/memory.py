import random
import numpy as np

from segment import SumTree, MinTree
from utils import Transition

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity        
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, alpha):
        super().__init__(capacity)

        tree_capacity = 1
        while tree_capacity < self.capacity:
            tree_capacity *= 2

        self.alpha = alpha
        self.max_priority = 1.0
        self.tree_ptr = 0

        self.sum_tree = SumTree(tree_capacity)
        self.min_tree = MinTree(tree_capacity)

    def push(self, *args):
        super().push(*args)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

    def sample(self, batch_size, beta):
        indices = self.sample_hook(batch_size)
        transitions = [self.memory[_] for _ in indices]
        weights = np.array([self.calculate_weight(_, beta) for _ in indices])

        return transitions, weights, indices
        
    def sample_hook(self, batch_size):
        indices = []
        total = self.sum_tree.sum(0, len(self) - 1)
        seg = total / batch_size

        for i in range(batch_size):
            l, r = seg * i, seg * (i + 1)
            upper = random.uniform(l, r)
            idx = self.sum_tree.upper(upper)
            if idx >= len(self):
                print('leak :', idx)
                idx = len(self) - 1
            indices.append(idx)

        return indices

    def update(self, indices, pri):
        for idx, p in zip(indices, pri):
            self.sum_tree[idx] = p ** self.alpha
            self.min_tree[idx] = p ** self.alpha

            self.max_priority = max(self.max_priority, p)

    def calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum(0, len(self) - 1)
        max_weight = (p_min * len(self)) ** (-beta)
        
        p_sample = self.sum_tree[idx] / self.sum_tree.sum(0, len(self) - 1)
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
