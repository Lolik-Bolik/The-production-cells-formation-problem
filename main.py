import utils
from algorithms import AnnealingSimulated
import os
from time import time
import numpy as np


def main():
    args = {'num_clusters': 2,
            'T0': 5,
            'Tf': 1e-3,
            'cooling_rate': 0.7,
            'n_iter': 4,
            'period': 6,
            'trap_stag_limit': 1,
            'max_cell_number': 20}
    for filename in os.listdir('./cfp_data'):
        data = utils.CellsProductionData(os.path.join('./cfp_data', filename))
        # Create data matrix with machines and parts
        data()
        tic = time()
        ann_sim_method = AnnealingSimulated(data)
        T, objective,borders,  cell_num = ann_sim_method(**args)
        # TODO remove code duplicating
        with open(f'{filename[:-4]}.sol', 'w') as file:
            machines = np.arange(1, data.matrix.shape[0] + 1)
            clusters_for_machines = []
            for i in range(len(borders)):
                if i != len(borders) - 1:
                    low_border = borders[i][0]
                    high_border = borders[i+1][0]
                else:
                    low_border = borders[i][0]
                    high_border = data.matrix.shape[0]
                slice_size = range(low_border, high_border)
                for _ in slice_size:
                    clusters_for_machines.append(i+1)
            for machine_id, machine_cluster_id in zip(machines, clusters_for_machines):
                file.write(f'm{machine_id}_{machine_cluster_id} ')
            file.write('\n')
            parts = np.arange(1, data.matrix.shape[1] + 1)
            clusters_for_parts = []
            for i in range(len(borders)):
                if i != len(borders) - 1:
                    low_border = borders[i][1]
                    high_border = borders[i+1][1]
                else:
                    low_border = borders[i][1]
                    high_border = data.matrix.shape[1]
                slice_size = range(low_border, high_border)
                for _ in slice_size:
                    clusters_for_parts.append(i+1)
            for parts_id, parts_cluster_id in zip(parts, clusters_for_parts):
                file.write(f'p{parts_id}_{parts_cluster_id} ')
            file.write('\n')
    toc = time()
    print(f'Final T: {T}\tFinal Objective: {objective}\t CellNum: {cell_num}\t Time {toc - tic}')

if __name__ == '__main__':
    main()