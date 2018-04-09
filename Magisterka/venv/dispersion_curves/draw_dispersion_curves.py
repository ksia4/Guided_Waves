from functions import calculate_dispersion_curves
from functions import draw_dispersion_curves


def main():
    if __name__ == "__main__":

        Fsys, C, ndof, kvect = calculate_dispersion_curves()
        draw_dispersion_curves(C, Fsys, kvect, ndof)


main()