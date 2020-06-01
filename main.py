import utils
from algorithms import AnnealingSimulated

def main():
    data = utils.CellsProductionData('./cfp_data/20x20.txt')
    # Create data matrix with machines and parts
    data()
    ann_sim_method = AnnealingSimulated(data)
    args = {'num_clusters': 2,
            'T0': 1,
            'Tf': 0.0001,
            'cooling_rate': 0.7,
            'n_iter': 5,
            'period': 6,
            'trap_stag_limit': 4}
    ann_sim_method(**args)
    print(ann_sim_method.target_value)

if __name__ == '__main__':
    main()