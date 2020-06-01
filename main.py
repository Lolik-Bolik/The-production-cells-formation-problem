import utils
from algorithms import AnnealingSimulated
import numpy as np

def main():
    data = utils.CellsProductionData('./cfp_data/20x20.txt')
    # Create data matrix with machines and parts
    data()
    ann_sim_method = AnnealingSimulated(data)
    sim_m, sim_p = ann_sim_method.similar_measure()
    cells_border = ann_sim_method.generate_solution_for_parts(sim_m, 2)
    ann_sim_method.generate_solution_by_machines(cells_border)


if __name__ == '__main__':
    main()