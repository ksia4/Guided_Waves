import sympy as sp
import numpy as np
from MES_dir import MES, config, dispersion


x, y, z = sp.symbols('x, y, z')

if __name__ == "__main__":
    print("Wpisz wartość: ")
    print("1 - rysowanie krzywy dyspersji z wykorzystaniem MES")
    print("2 - rysowanie krzywych dyspersji z ostatnio policzonych danych")
    text = input()

    if text == '1':
        # parametry preta
        length = 3
        radius = 10
        num_of_circles = 6
        num_of_points_at_c1 = 6

        # wektor liczby falowej
        config.kvect_min = 1e-10
        config.kvect_max = np.pi / 2
        config.kvect_no_of_points = 101

        # rysowanie wykresow
        config.show_plane = False
        config.show_bar = False
        config.show_elements = False

        # obliczenia
        MES.mes(length, radius, num_of_circles, num_of_points_at_c1)
        # print(np.shape(config.k))
        # print(np.shape(config.m))
        a = 5
        dispersion.draw_dispercion_curves()
        print("koniec")


    # rysowanie krzywych dyspersji z wczesniej obliczonych wartosci
    if text == '2':
        dispersion.draw_dispercion_curves_from_file()
