class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0. for _ in range(2 * capacity)]

    def sum_hook(self, left, right, node, node_left, node_right):
        if left <= node_left and node_right <= right:
            return self.tree[node]

        result = 0
        mid = (node_left + node_right) // 2
        
        if left <= mid:
            result += self.sum_hook(left, right, node * 2, node_left, mid)
        if right > mid:
            result += self.sum_hook(left, right, node * 2 + 1, mid + 1, node_right)

        return result

    def sum(self, left, right):
        return self.sum_hook(left, right, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2
        
    def __getitem__(self, idx):
        return self.tree[self.capacity + idx]

    def upper(self, val):
        idx = 1

        while idx < self.capacity:
            left = idx * 2
            if self.tree[left] > val:
                idx = idx * 2
            else:
                val -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

class MinTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [float('inf') for _ in range(2 * capacity)]
        
    def min(self):
        return self.tree[1]

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2
        
    def __getitem__(self, idx):
        return self.tree[self.capacity + idx]
