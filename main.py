import utils


def main():
    data = utils.CellsProductionData('./cfp_data/20x20.txt')
    # Create data matrix with machines and parts
    data()


if __name__ == '__main__':
    main()