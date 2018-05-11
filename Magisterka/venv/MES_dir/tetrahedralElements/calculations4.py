import sympy as sp
import numpy as np
import numpy.linalg as la
import scipy.integrate as integr
from MES_dir.tetrahedralElements import gauss4


# TODO: wprowadzenie sily wymuszajacej i warunkow brzegowych
# TODO: rozwiazanie ukladu - metoda roznicowa
# TODO: identyfikacja punktow w elementach

# UWAGA! Wszystko dla 4-wierzchołkowych elementów 3D

def p_vector():
    x, y, z = sp.symbols('x, y, z')
    p = [1, x, y, z]
    return sp.Matrix(p).reshape(1, 4)

# indicies - indeksy wezlow jednego elementu (jeden wiersz z Delaunay-a)
# verticies - macierz punktow siatki
def me_matrix(vertices, element_indices):
    me = []
    for ind in element_indices:
        temp = [1, vertices[ind, 0], vertices[ind, 1], vertices[ind, 2]]
        me.append(temp)
    return sp.Matrix(me)


def me_inv_matrix(vertices, element_indices):
    me = []
    for ind in element_indices:
        temp = [1, vertices[ind, 0], vertices[ind, 1], vertices[ind, 2]]
        me.append(temp)
    return sp.Matrix(me).inv()


def shape_functions(vertices, element_indices):
    x, y, z = sp.symbols('x, y, z')
    p = p_vector()
    me_inv = me_inv_matrix(vertices, element_indices)
    return p * me_inv  # N


def b_matrix_fc(shape_functions):
    x, y, z = sp.symbols('x, y, z')
    b = []
    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []
    w6 = []

    for n in shape_functions:
        w1.append(sp.diff(n, x))
        w1.append(0)
        w1.append(0)

        w2.append(0)
        w2.append(sp.diff(n, y))
        w2.append(0)

        w3.append(0)
        w3.append(0)
        w3.append(sp.diff(n, z))

        w4.append(sp.diff(n, y))
        w4.append(sp.diff(n, x))
        w4.append(0)

        w5.append(sp.diff(n, z))
        w5.append(0)
        w5.append(sp.diff(n, x))

        w6.append(0)
        w6.append(sp.diff(n, z))
        w6.append(sp.diff(n, y))

    b.append(w1)
    b.append(w2)
    b.append(w3)
    b.append(w4)
    b.append(w5)
    b.append(w6)

    return np.array(b)


def b_matrix_natural(shape_functions):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    b = []
    w1 = []
    w2 = []
    w3 = []
    w4 = []
    w5 = []
    w6 = []

    for n in shape_functions:
        w1.append(sp.diff(n, ksi))
        w1.append(0)
        w1.append(0)

        w2.append(0)
        w2.append(sp.diff(n, eta))
        w2.append(0)

        w3.append(0)
        w3.append(0)
        w3.append(sp.diff(n, dzeta))

        w4.append(sp.diff(n, eta))
        w4.append(sp.diff(n, ksi))
        w4.append(0)

        w5.append(sp.diff(n, dzeta))
        w5.append(0)
        w5.append(sp.diff(n, ksi))

        w6.append(0)
        w6.append(sp.diff(n, dzeta))
        w6.append(sp.diff(n, eta))

    b.append(w1)
    b.append(w2)
    b.append(w3)
    b.append(w4)
    b.append(w5)
    b.append(w6)

    return np.array(b)


def d_matrix_fc(young_modulus, poisson_coeficient):
    a = young_modulus / ((1 + poisson_coeficient) * (1 - 2*poisson_coeficient))
    b1 = (1 - poisson_coeficient) * a
    b2 = (poisson_coeficient) * a
    b3 = ((1 - 2*poisson_coeficient) / 2) * a
    matrix = [[b1, b2, b2, 0, 0, 0], [b2, b1, b2, 0, 0, 0], [b2, b2, b1, 0, 0, 0],
              [0, 0, 0, b3, 0, 0], [0, 0, 0, 0, b3, 0], [0, 0, 0, 0, 0, b3]]
    return np.array(matrix)


def stiff_local_matrix(shape_functions, vertices, element_indices, young_modulus, poisson_coefficient):

    element_vertices = vertices[element_indices, :]

    v = volume_det(element_vertices)

    b_matrix = b_matrix_fc(shape_functions)
    d_matrix = d_matrix_fc(young_modulus, poisson_coefficient)

    b_tran = np.transpose(b_matrix)

    temp = np.dot(b_tran, d_matrix)
    temp1 = np.dot(temp, b_matrix)

    stiff = temp1 * v

    return np.array(stiff)


def mass_local_matrix(density):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

    N_integrate = gauss4.matrix_to_integrate(density)
    #granice calkowania
    y_bot = lambda ksi: -1
    y_top = lambda ksi: -ksi
    z_bot = lambda ksi, eta: -1
    z_top = lambda ksi, eta: -ksi - eta - 1

    integral = []

    for row in N_integrate:
        integr_row = []
        for elem in row:
            if elem != 0:
                lam_elem = sp.lambdify((ksi, eta, dzeta), elem)

                temp = integr.tplquad(lam_elem, -1, 1, y_bot, y_top, z_bot, z_top)[0]

            if elem == 0:
                temp = 0

            integr_row.append(temp)
        integral.append(integr_row)

    return np.array(integral)


#geometrycznie
def volume(element_vertices):
    #dlugosci bokow podstawy
    d1 = la.norm(element_vertices[0] - element_vertices[1])
    d2 = la.norm(element_vertices[0] - element_vertices[2])
    d3 = la.norm(element_vertices[1] - element_vertices[2])
    p = (d1 + d2 + d3)/2
    #pole podstawy - wzor Herona
    area = np.sqrt(p*(p-d1)*(p-d2)*(p-d3))

    # poszukiwanie rownania plaszczyzny podstawy
    a = np.array([list(element_vertices[0]), list(element_vertices[1]), list(element_vertices[2])])

    b = np.array([-1, -1, -1])
    wsp = la.solve(a, b)

    #obliczanie wysokosci - odleglosc punktu do plaszczyzny
    numerator = abs(wsp[0] * element_vertices[3, 0] + wsp[1] * element_vertices[3, 1] + wsp[2] * element_vertices[3, 2] + 1)
    denominator = np.sqrt(wsp[0]**2 + wsp[1]**2 + wsp[2]**2)
    h = numerator/denominator

    #objetosc ostroslupa
    volume = (1/3)*area*h
    return volume


#wyznacznikiem
def volume_det(vertices):
    #to samo wyznacznikiem
    temp1 = [1]
    temp1.extend(list(vertices[0]))
    temp2 = [1]
    temp2.extend(list(vertices[1]))
    temp3 = [1]
    temp3.extend(list(vertices[2]))
    temp4 = [1]
    temp4.extend(list(vertices[3]))

    a = [temp1, temp2, temp3, temp4]
    a1 = np.array(a)

    #dzielone przez 6 zeby wyszlo jak w drugiej funkcji - czy dobrze???
    return abs(la.det(a1)/6)



