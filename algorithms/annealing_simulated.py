import numpy as np
from itertools import combinations, permutations


class AnnealingSimulated:
    def __init__(self, data, temperature=0.8):
        self.matrix = data.matrix
        self.temperature = temperature
        self.machines_amount = data.machines_amount
        self.parts_amount = data.parts_amount

    def similar_measure(self):
        similarity_matrix = np.negative(np.ones((self.parts_amount, self.parts_amount)))
        similarity_pairs = {}
        for i, j in combinations(range(0, self.parts_amount), 2):
            a_ij, b_ij, c_ij = 0, 0, 0
            for machine_number in range(self.machines_amount):
                if self.matrix[machine_number][i] and self.matrix[machine_number][j]:
                    a_ij += 1
                elif self.matrix[machine_number][i] and not self.matrix[machine_number][j]:
                    b_ij += 1
                elif not self.matrix[machine_number][i] and self.matrix[machine_number][j]:
                    c_ij += 1
            similarity_coeff = a_ij / (a_ij + b_ij + c_ij)
            similarity_pairs[(i, j)] = similarity_coeff
            similarity_pairs[(j, i)] = similarity_coeff
            similarity_matrix[i][j] = similarity_coeff
            similarity_matrix[j][i] = similarity_coeff
        similarity_pairs = sorted(similarity_pairs.items(), key=lambda kv: -kv[1])
        return similarity_matrix, similarity_pairs

    def update_temperature(self, a):
        self.temperature *= a

