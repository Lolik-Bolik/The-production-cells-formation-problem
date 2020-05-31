import numpy as np


class CellsProductionData:
    def __init__(self, path_to_data):
        self.path = path_to_data
        self.matrix = None
        self.machines_amount = None
        self.parts_amount = None

    def __call__(self, *args, **kwargs):
        with open(self.path, "r") as f:
            m, n = map(int, f.readline().split())
            self.machines_amount = m
            self.parts_amount = n
            self.matrix = np.empty((self.machines_amount, self.parts_amount), dtype=int)
            for i in range(self.machines_amount):
                for j in list(map(int, f.readline().split()))[1:]:
                    self.matrix[i][j - 1] = 1

