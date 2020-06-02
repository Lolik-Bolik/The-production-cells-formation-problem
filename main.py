import utils
from algorithms import AnnealingSimulated
import os
from time import time

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
        print(filename)
        data = utils.CellsProductionData(os.path.join('./cfp_data', filename))
        # Create data matrix with machines and parts
        data()
        tic = time()
        ann_sim_method = AnnealingSimulated(data)
        T, objective, cell_num = ann_sim_method(**args)
        toc = time()
        print(f'Final T: {T}\tFinal Objective: {objective}\t CellNum: {cell_num}\t Time {toc - tic}')

if __name__ == '__main__':
    main()