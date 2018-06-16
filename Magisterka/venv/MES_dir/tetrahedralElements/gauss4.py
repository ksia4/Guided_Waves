import numpy as np
import numpy.linalg as la
import sympy as sp

# Calkowanie odbywa sie juz w calculations/mass_local_matrix

#Funkcje kształtu we współrzędnych naturalnych.
def shape_functions_natural():
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

    N = [-0.5 - 0.5 * ksi - 0.5 * eta - 0.5 * dzeta,
         0.5 + 0.5 * ksi,
         0.5 + 0.5 * eta,
         0.5 + 0.5 * dzeta]

    return N

#Wyznaczanie współrzędnych rzeczywistych w funkcji współrzędnych naturalnych.
def coordinate_change_model(vertices, natural_shape_fc):

    x = vertices[0, 0]*natural_shape_fc[0] + vertices[1, 0]*natural_shape_fc[1] +\
        vertices[2, 0]*natural_shape_fc[2] + vertices[3, 0]*natural_shape_fc[3]
    y = vertices[0, 1]*natural_shape_fc[0] + vertices[1, 1]*natural_shape_fc[1] +\
        vertices[2, 1]*natural_shape_fc[2] + vertices[3, 1]*natural_shape_fc[3]
    z = vertices[0, 2]*natural_shape_fc[0] + vertices[1, 2]*natural_shape_fc[1] +\
        vertices[2, 2]*natural_shape_fc[2] + vertices[3, 2]*natural_shape_fc[3]

    return x, y, z

#Wyznacza jakobian przekształcenia.
def jacobian(vertices, natural_shape_fc):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    x_in_nat, y_in_nat, z_in_nat = coordinate_change_model(vertices, natural_shape_fc)
    j = sp.Matrix([[sp.diff(x_in_nat, ksi), sp.diff(y_in_nat, ksi), sp.diff(z_in_nat, ksi)],
                    [sp.diff(x_in_nat, eta), sp.diff(y_in_nat, eta), sp.diff(z_in_nat, eta)],
                   [sp.diff(x_in_nat, dzeta), sp.diff(y_in_nat, dzeta), sp.diff(z_in_nat, dzeta)]])

    return sp.det(j) # jacobian

#Wyznacza macierz podcałkową dla macierzy mas.
def matrix_to_integrate(density):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    shape_functions = shape_functions_natural()

    N = np.array([[shape_functions[0], 0, 0, shape_functions[1], 0, 0,
                   shape_functions[2], 0, 0, shape_functions[3], 0, 0],
         [0, shape_functions[0], 0, 0, shape_functions[1], 0,
          0, shape_functions[2], 0, 0, shape_functions[3], 0],
         [0, 0, shape_functions[0], 0, 0, shape_functions[1],
          0, 0, shape_functions[2], 0, 0, shape_functions[3]]])

    # Macierz podcalkowa
    N_integrate = N.transpose().dot(N)*density

    return N_integrate


