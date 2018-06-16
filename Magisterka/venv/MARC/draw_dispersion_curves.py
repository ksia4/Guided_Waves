from MARC.functions import calculate_dispersion_curves
from MARC.functions import draw_dispersion_curves


def main():
    if __name__ == "__main__":

        Fsys, C, ndof, kvect = calculate_dispersion_curves()
        draw_dispersion_curves(C, Fsys, kvect, ndof)


main()