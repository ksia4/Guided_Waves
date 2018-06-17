import numpy as np
import numpy.linalg as la
import sympy as sp

# Calkowanie odbywa sie juz w calculations/mass_local_matrix

#Funkcje kształtu we współrzędnych naturalnych.
def shapeFunctionsNatural():
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

    N = [-0.5 - 0.5 * ksi - 0.5 * eta - 0.5 * dzeta,
         0.5 + 0.5 * ksi,
         0.5 + 0.5 * eta,
         0.5 + 0.5 * dzeta]

    return N

#Wyznaczanie współrzędnych rzeczywistych w funkcji współrzędnych naturalnych.
def coordinateChangeModel(elementVertices, naturalShapeFc):

    x = elementVertices[0, 0] * naturalShapeFc[0] + elementVertices[1, 0] * naturalShapeFc[1] + \
        elementVertices[2, 0] * naturalShapeFc[2] + elementVertices[3, 0] * naturalShapeFc[3]
    y = elementVertices[0, 1] * naturalShapeFc[0] + elementVertices[1, 1] * naturalShapeFc[1] + \
        elementVertices[2, 1] * naturalShapeFc[2] + elementVertices[3, 1] * naturalShapeFc[3]
    z = elementVertices[0, 2] * naturalShapeFc[0] + elementVertices[1, 2] * naturalShapeFc[1] + \
        elementVertices[2, 2] * naturalShapeFc[2] + elementVertices[3, 2] * naturalShapeFc[3]

    return x, y, z

#Wyznacza jakobian przekształcenia.
def jacobian(elementVertices, naturalShapeFc):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    x_in_nat, y_in_nat, z_in_nat = coordinateChangeModel(elementVertices, naturalShapeFc)
    j = sp.Matrix([[sp.diff(x_in_nat, ksi), sp.diff(y_in_nat, ksi), sp.diff(z_in_nat, ksi)],
                    [sp.diff(x_in_nat, eta), sp.diff(y_in_nat, eta), sp.diff(z_in_nat, eta)],
                   [sp.diff(x_in_nat, dzeta), sp.diff(y_in_nat, dzeta), sp.diff(z_in_nat, dzeta)]])

    return sp.det(j) # jacobian

#Wyznacza macierz podcałkową dla macierzy mas.
def matrixToIntegrate(density):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    shape_functions = shapeFunctionsNatural()

    N = np.array([[shape_functions[0], 0, 0, shape_functions[1], 0, 0,
                   shape_functions[2], 0, 0, shape_functions[3], 0, 0],
         [0, shape_functions[0], 0, 0, shape_functions[1], 0,
          0, shape_functions[2], 0, 0, shape_functions[3], 0],
         [0, 0, shape_functions[0], 0, 0, shape_functions[1],
          0, 0, shape_functions[2], 0, 0, shape_functions[3]]])

    # Macierz podcalkowa
    N_integrate = N.transpose().dot(N)*density

    return N_integrate


