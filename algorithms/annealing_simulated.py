import numpy as np
from itertools import combinations, permutations
import scipy.cluster.hierarchy as spc

'''np.asarray([[1,0,0,1,0],
                       [0,1,1,0,1],
                       [1,0,0,0,0],
                       [0,1,1,0,0],
                       [0,0,0,1,0]])'''


class AnnealingSimulated:
    def __init__(self, data, temperature=0.8):
        self.matrix = data.matrix
        self.temperature = temperature
        self.machines_amount = data.machines_amount
        self.parts_amount = data.parts_amount

    def similar_measure(self):
        similarity_matrix = np.ones((self.parts_amount, self.parts_amount))
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

    def generate_solution_for_parts(self, similarity_matrix, num_clusters):
        pdist = spc.distance.pdist(similarity_matrix)
        linkage = spc.linkage(pdist, method='complete')
        parts_cells = spc.fcluster(linkage, num_clusters, 'maxclust')
        idx_sort = np.argsort(parts_cells)
        sorted_records_array = parts_cells[idx_sort]
        _, cells_border = np.unique(sorted_records_array, return_index=True)
        self.matrix = self.matrix[..., idx_sort]
        return cells_border

    def generate_solution_by_machines(self, cells_border):
        machines_matrix = np.zeros((self.machines_amount, len(cells_border)))
        for i in range(self.machines_amount):
            for j in range(len(cells_border)):
                if j == len(cells_border) -1:
                    low_border = cells_border[j]
                    high_border = None
                else:
                    low_border = cells_border[j]
                    high_border = cells_border[j+1]
                machine = self.matrix[i, low_border:high_border]
                voids = np.count_nonzero(self.matrix[i]) - np.count_nonzero(machine)
                exceptional = machine.size - np.count_nonzero(machine)
                machines_matrix[i][j] = voids + exceptional
        indexes = np.argmin(machines_matrix, axis=1)
        idx_sort = np.argsort(indexes)
        sorted_records_array = indexes[idx_sort]
        _, cells_border_machines = np.unique(sorted_records_array, return_index=True)
        self.matrix = self.matrix[idx_sort, ...]
        cells_border = np.column_stack((cells_border_machines, cells_border))
        return cells_border






