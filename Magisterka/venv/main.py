import sympy as sp
import numpy as np
from MES_dir import MES, config, dispersion_curves
from MARC import functions

# begin MES
x, y, z = sp.symbols('x, y, z')

if __name__ == "__main__":
    print("Wpisz wartość: ")
    print("1 - rysowanie krzywy dyspersji z wykorzystaniem MES")
    print("2 - rysowanie krzywych dyspersji z ostatnio policzonych danych")
    text = input()

    if text == '1':

        print("Wpisz wartość: ")
        print("4 - elementy czworościenne")
        print("8 - elemnty sześcienne")
        print("M - wczytanie macierzy z MARC i wykreślenie krzywych")
        text1 = input()

        # wektor liczby falowej
        config.kvect_min = 1e-10
        config.kvect_max = np.pi / 4
        config.kvect_no_of_points = 301

        # rysowanie wykresow
        config.show_plane = True
        config.show_bar = False
        config.show_elements = False
		
		# zapisywanie wektorow wlasnych
		saveEigVectors = True


        # obliczenia
        if text1 == '4':
            # parametry preta
            radius = 25
            num_of_circles = 8
            num_of_points_at_c1 = 8
            MES.mes4(radius, num_of_circles, num_of_points_at_c1)

        if text1 == '8':
            radius = 25
            numberOfPlanes = 3
            firstCircle = 16    # for brickMesh should be 16
            addNodes = 0    # for brickMesh doesn't matter
            circles = 10
            MES.mes8(numberOfPlanes, radius, circles, firstCircle, addNodes)

        if text1 == 'M':
            config.k, config.m = functions.getStiffAndMassMatrix()

        dispersion_curves.drawDispercionCurves()
        print("koniec")

    # rysowanie krzywych dyspersji z wczesniej obliczonych wartosci
    if text == '2':
        dispersion_curves.drawDispercionCurvesFromFile()

