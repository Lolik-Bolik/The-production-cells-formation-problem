import numpy as np
from itertools import combinations, permutations
import scipy.cluster.hierarchy as spc
import random
from math import exp

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
        self.target_value = -1

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
        n_1_out = n_1 - n_1_in
        # return n_1_in / (n_1 + n_0_in)
        return (n_1 - n_1_out) / (n_1 + n_0_in)

    def generate_solution_for_parts(self, similarity_matrix, num_clusters):
        pdist = spc.distance.pdist(similarity_matrix)
        linkage = spc.linkage(pdist, method='complete')
        cell_clusters = spc.fcluster(linkage, num_clusters, 'maxclust')
        idx_sort = np.argsort(cell_clusters)
        sorted_cell_clusters = cell_clusters[idx_sort]
        _, cell_borders = np.unique(sorted_cell_clusters, return_index=True)
        self.matrix = self.matrix[..., idx_sort]
        return cell_borders

    def generate_solution_by_machines(self, matrix=None, cell_borders=None, save=True):
        if matrix is None:
            matrix = self.matrix
        if cell_borders is None:
            cell_borders = self.cell_borders
        if len(cell_borders.shape) == 2:
            cell_borders = cell_borders[..., 1]
        machines_matrix = np.zeros((self.machines_amount, len(cell_borders)))
        for j in range(len(cell_borders)):
            if j == len(cell_borders) - 1:
                low_border = cell_borders[j]
                high_border = None
            else:
                low_border = cell_borders[j]
                high_border = cell_borders[j+1]
            machine = matrix[..., low_border:high_border]
            voids = np.count_nonzero(matrix, axis=1) - np.count_nonzero(machine, axis=1)
            exceptional = machine.shape[1] - np.count_nonzero(machine, axis=1)
            machines_matrix[..., j] = voids + exceptional
        cell_clusters = np.argmin(machines_matrix, axis=1)
        idx_sort = np.argsort(cell_clusters)
        sorted_cell_clusters = cell_clusters[idx_sort]
        _, cell_borders_machines = np.unique(sorted_cell_clusters, return_index=True)
        while cell_borders_machines.shape != cell_borders.shape:
            cell_borders_machines = np.append(cell_borders_machines, self.matrix.shape[0])
        if save:
            self.matrix = matrix[idx_sort, ...]
            self.cell_borders = np.column_stack((cell_borders_machines, cell_borders))
        return matrix[idx_sort, ...], np.column_stack((cell_borders_machines, cell_borders))

    def single_move(self):
        matrix = self.matrix.copy()
        best_objective_value = None
        best_matrix = None
        best_borders = None
        overload = False
        # Пробегаем по всем границам
        for idx in range(len(self.cell_borders)):
            # Если на последней итерации, берем last + 1 или matrix.shape, если это не последний столбец
            if idx == len(self.cell_borders) - 1:
                low_border = self.cell_borders[idx][1]
                if low_border == matrix.shape[1]:
                    high_border = low_border + 1
                    overload = True
                else:
                    high_border = matrix.shape[1]
                    overload = True
            else:
                low_border = self.cell_borders[idx][1]
                high_border = self.cell_borders[idx + 1][1]
            # Создаем столбцы текущей клетки
            source_iter_range = range(low_border, high_border)
            # Берем НИЖНИЕ границы других клеток, кроме той клетки, в которой сейчас находимся
            target_positions = [(k, border) for k, border in enumerate(self.cell_borders) if (border != self.cell_borders[idx]).all()]
            # Итерируем по столбцам текущей клетки
            for source_position in source_iter_range:
                source_part = matrix[..., source_position]
                for k, target_position in target_positions:
                    # Вставляем текущий столбец перед началом таргетной клетки (на ее нижнюю границу)
                    tmp_matrix = np.insert(matrix, target_position[1], source_part, axis=1)
                    tmp_borders = self.cell_borders.copy()
                    # Если столбец прыгает направо
                    if source_position < target_position[1]:
                        # Уменьшаем нижнюю границу таргетной клетки на 1
                        try:
                            tmp_borders[idx + 1][1] -= 1
                        except:
                            pass
                        # В случае нескольких кластеров нужно еще уменьшить верхнюю границу источника на 1, если источник и таргет несмежны
                        if idx + 1 != k:
                            tmp_borders[k][1] -= 1
                        tmp_matrix = np.delete(tmp_matrix, source_position, axis=1)
                    # Если столбец прыгает налево
                    else:
                        # Увеличиваем нижнюю границу текущей клетки на 1
                        tmp_borders[idx][1] += 1
                        # В случае нескольких кластеров нужно еще увеличить верхнюю границу таргета на 1, если источник и таргет несмежны
                        if idx != k + 1:
                            try:
                                tmp_borders[k + 1][1] += 1
                            except:
                                pass
                        tmp_matrix = np.delete(tmp_matrix, source_position + 1, axis=1)
                    objective_value = self.calculate_target_value(tmp_matrix, tmp_borders)
                    if overload:
                        tmp_borders[-1, 1] -= 1
                    if best_objective_value is None:
                        best_objective_value = objective_value
                        best_matrix = tmp_matrix
                        best_borders = tmp_borders
                    elif (objective_value - self.target_value) > (best_objective_value - self.target_value):
                        best_objective_value = objective_value
                        best_matrix = tmp_matrix
                        best_borders = tmp_borders

        return best_matrix, best_borders

    def exchange_move(self):
        matrix = self.matrix.copy()
        best_objective_value = None
        best_matrix = None
        for idx in range(len(self.cell_borders)):
            if idx == len(self.cell_borders) - 1:
                low_border = self.cell_borders[idx][1]
                if low_border == matrix.shape[1]:
                    high_border = low_border + 1
                else:
                    high_border = matrix.shape[1]
            else:
                low_border = self.cell_borders[idx][1]
                high_border = self.cell_borders[idx + 1][1]
            source_iter_range = range(low_border, high_border)
            target_cells_indexes = [k for k, border in enumerate(self.cell_borders) if
                                (border != self.cell_borders[idx]).all()]
            for source_position in source_iter_range:
                for k in target_cells_indexes:
                    if k == len(self.cell_borders) - 1:
                        low_target_border = self.cell_borders[k][1]
                        if low_target_border == matrix.shape[1]:
                            high_target_border = low_target_border + 1
                        else:
                            high_target_border = matrix.shape[1]
                    else:
                        low_target_border = self.cell_borders[k][1]
                        high_target_border = self.cell_borders[k + 1][1]
                    target_iter_range = range(low_target_border, high_target_border)
                    for target_position in target_iter_range:
                        tmp_matrix = matrix.copy()
                        tmp_matrix[..., [source_position, target_position]] = tmp_matrix[..., [target_position, source_position]]
                        objective_value = self.calculate_target_value(tmp_matrix, self.cell_borders)
                        if best_objective_value is None:
                            best_objective_value = objective_value
                            best_matrix = tmp_matrix
                        elif (objective_value - self.target_value) > (best_objective_value - self.target_value):
                            best_objective_value = objective_value
                            best_matrix = tmp_matrix
        return best_matrix, self.cell_borders
    
    def generate_initial_solution(self, cell_number):
        sim_m, sim_p = self.similar_measure()
        cell_borders = self.generate_solution_for_parts(sim_m, cell_number)
        self.generate_solution_by_machines(cell_borders=cell_borders)
        self.target_value = self.calculate_target_value()
        best_sol_at_curr_cell = self.matrix.copy()
        best_current_cell_borders = self.cell_borders.copy()
        return best_sol_at_curr_cell, best_current_cell_borders


    def __call__(self, *args, **kwargs):
        cell_number = kwargs['num_clusters']
        optimal_cell_number = cell_number
        T_f = kwargs['Tf']
        cool_rate = kwargs['cooling_rate']
        n_iter = kwargs['n_iter']
        period = kwargs['period']
        trap_stag_limit = kwargs['trap_stag_limit']
        max_cell_number = kwargs['max_cell_number']
        T = kwargs['T0']
        counter = 0
        counter_MC = 0
        counter_trap = 0
        counter_stag = 0
        best_sol_at_all, best_at_all_borders = self.generate_initial_solution(cell_number)
        best_sol_at_curr_cell = best_sol_at_all.copy()
        best_current_cell_borders = best_at_all_borders.copy()
        best_at_all_target = self.calculate_target_value(best_sol_at_all, best_at_all_borders)
        print('Initial target: ', best_at_all_target)
        initial_sol = True
        generate_solution = True
        while(1):
            if initial_sol:
                initial_sol = False
            elif generate_solution:
                best_sol_at_curr_cell, best_current_cell_borders = self.generate_initial_solution(cell_number)
                generate_solution = False
            while counter_MC < n_iter and counter_trap < n_iter / 2:
                new_sol, new_borders = self.single_move()
                if counter % period == 0:
                    self.exchange_move()
                neighborhood_sol, neigbourhood_borders = self.generate_solution_by_machines(new_sol, new_borders, save=False)
                neigbourhood_target_value = self.calculate_target_value(neighborhood_sol, neigbourhood_borders)
                if neigbourhood_target_value > self.calculate_target_value(best_sol_at_curr_cell, best_current_cell_borders):#self.target_value:
                    best_sol_at_curr_cell = neighborhood_sol.copy()
                    best_current_cell_borders = neigbourhood_borders.copy()
                    self.matrix = neighborhood_sol.copy()
                    self.cell_borders = neigbourhood_borders.copy()
                    counter_stag = 0
                    counter_MC += 1
                elif neigbourhood_target_value == self.target_value:
                    self.matrix = neighborhood_sol.copy()
                    self.cell_borders = neigbourhood_borders.copy()
                    counter_stag += 1
                    counter_MC += 1
                else:
                    delta = neigbourhood_target_value - self.target_value
                    prob = random.random()

                    if exp(-delta/T) > prob:
                        self.matrix = neighborhood_sol.copy()
                        self.cell_borders = neigbourhood_borders.copy()
                        counter_trap = 0
                    else:
                        counter_trap += 1
                    counter_MC += 1
            if T < T_f or counter_stag >= trap_stag_limit:
                if self.calculate_target_value(best_sol_at_curr_cell, best_current_cell_borders) > self.calculate_target_value(best_sol_at_all, best_at_all_borders)\
                   and cell_number < max_cell_number:
                        best_sol_at_all = best_sol_at_curr_cell.copy()
                        best_at_all_borders = best_current_cell_borders.copy()
                        best_at_all_target = self.calculate_target_value(best_sol_at_all, best_at_all_borders)
                        optimal_cell_number = cell_number
                        cell_number += 1
                        T = kwargs['T0']
                        generate_solution = True
                        counter = 0
                        counter_MC = 0
                        counter_trap = 0
                        counter_stag = 0
                else:
                    return T, best_at_all_target, best_at_all_borders, optimal_cell_number
            else:
                T *= cool_rate
                counter_MC = 0
                counter_MC += 1
                counter += 1











