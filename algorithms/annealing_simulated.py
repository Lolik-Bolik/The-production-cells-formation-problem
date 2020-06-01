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
        self.cell_borders = None
        self.current_object_value = -1

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

    def calculate_target_value(self, matrix=None, cell_borders=None):
        if matrix is None:
            matrix = self.matrix
        if cell_borders is None:
            cell_borders = self.cell_borders
        n_1 = np.count_nonzero(matrix)
        n_1_in = 0
        n_0_in = 0
        for j in range(len(cell_borders)):
            if j == len(cell_borders) - 1:
                low_border = cell_borders[j]
                high_border = [None, None]
            else:
                low_border = cell_borders[j]
                high_border = cell_borders[j + 1]
            cell = matrix[low_border[0]:high_border[0], low_border[1]:high_border[1]]
            n_1_in += np.count_nonzero(cell)
            n_0_in += cell.size - np.count_nonzero(cell)
        return n_1_in / (n_1 + n_0_in)

    def generate_solution_for_parts(self, similarity_matrix, num_clusters):
        pdist = spc.distance.pdist(similarity_matrix)
        linkage = spc.linkage(pdist, method='complete')
        cell_clusters = spc.fcluster(linkage, num_clusters, 'maxclust')
        idx_sort = np.argsort(cell_clusters)
        sorted_cell_clusters = cell_clusters[idx_sort]
        _, cell_borders = np.unique(sorted_cell_clusters, return_index=True)
        self.matrix = self.matrix[..., idx_sort]
        return cell_borders

    def generate_solution_by_machines(self, cell_borders):
        # TODO: Падает на некоторых количествах кластерах и всегда падает на неквадратных данных
        machines_matrix = np.zeros((self.machines_amount, len(cell_borders)))
        for j in range(len(cell_borders)):
            if j == len(cell_borders) - 1:
                low_border = cell_borders[j]
                high_border = None
            else:
                low_border = cell_borders[j]
                high_border = cell_borders[j+1]
            machine = self.matrix[..., low_border:high_border]
            voids = np.count_nonzero(self.matrix, axis=1) - np.count_nonzero(machine, axis=1)
            exceptional = machine.shape[1] - np.count_nonzero(machine, axis=1)
            machines_matrix[..., j] = voids + exceptional
        cell_clusters = np.argmin(machines_matrix, axis=1)
        idx_sort = np.argsort(cell_clusters)
        sorted_cell_clusters = cell_clusters[idx_sort]
        _, cell_borders_machines = np.unique(sorted_cell_clusters, return_index=True)
        self.matrix = self.matrix[idx_sort, ...]
        self.cell_borders = np.column_stack((cell_borders_machines, cell_borders))
        return cell_borders

    def single_move(self):
        # TODO: Возможно, забагован. Также ЗАКОММЕНТИТЬ, а то нихрена не понятно
        matrix = self.matrix.copy()
        best_objective_value = None
        best_matrix = None
        best_borders = None
        for idx in range(len(self.cell_borders)):
            if idx == len(self.cell_borders) - 1:
                low_border = self.cell_borders[idx][1]
                high_border = low_border + 1 if low_border == matrix.shape[1] else matrix.shape[1]
            else:
                low_border = self.cell_borders[idx][1]
                high_border = self.cell_borders[idx + 1][1]
            source_iter_range = range(low_border, high_border)
            target_positions = [(k, border) for k, border in enumerate(self.cell_borders) if (border != self.cell_borders[idx]).all()]
            for source_position in source_iter_range:
                source_part = matrix[..., source_position]
                for k, target_position in target_positions:
                    tmp_matrix = np.insert(matrix, target_position[1], source_part, axis=1)
                    tmp_borders = self.cell_borders.copy()
                    if source_position < target_position[1]:
                        tmp_borders[idx + 1][1] -= 1
                        tmp_matrix = np.delete(tmp_matrix, source_position, axis=1)
                    else:
                        tmp_borders[idx][1] += 1
                        tmp_matrix = np.delete(tmp_matrix, source_position + 1, axis=1)
                    objective_value = self.calculate_target_value(tmp_matrix, tmp_borders)
                    if best_objective_value is None:
                        best_objective_value = objective_value
                        best_matrix = tmp_matrix
                        best_borders = tmp_borders
                    elif (objective_value - self.current_object_value) > (best_objective_value - self.current_object_value):
                        best_objective_value = objective_value
                        best_matrix = tmp_matrix
                        best_borders = tmp_borders
        self.matrix = best_matrix
        self.cell_borders = best_borders
        self.current_object_value = best_objective_value









