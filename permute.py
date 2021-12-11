import torch as t


class Permute:
    def __init__(self, array):
        self.array = array

    def permute(self):
        a = t.from_numpy(self.array).float()
        return a.permute(0, 4, 1, 2, 3)
