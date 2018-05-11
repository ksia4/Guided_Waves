import numpy as np
import numpy.linalg as la
import sympy as sp



# Calkowanie odbywa sie juz w calculations/mass_local_matrix

#nie uzywana
def change_to_natural_coordinates(shape_fuctions):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    x, y, z = sp.symbols('x, y, z')

    # wspolrzedne naturalne dla 4 punktow
    ksi_e = [-1, 1, -1, -1]
    eta_e = [-1, -1, 1, -1]
    dzeta_e = [-1, -1, -1, 1]

    # obliczanie wspolrzednych naturalnych w funkcji x, y, z
    ksi1 = ksi_e[0] * shape_fuctions[0] - ksi
    eta1 = eta_e[0] * shape_fuctions[0] - eta
    dzeta1 = dzeta_e[0] * shape_fuctions[0] -dzeta

    for i in range(1, 4):
        ksi1 += ksi_e[i] * shape_fuctions[i]
        eta1 += eta_e[i] * shape_fuctions[i]
        dzeta1 += dzeta_e[i] * shape_fuctions[i]

    # obliczanie x, y, z w funkcji wspolrzednych naturalnych
    list_of_eq = [ksi1, eta1, dzeta1]

    t = sp.linsolve(list_of_eq, x, y, z)
    (xx, yy, zz) = next(iter(t))

    shape_fuctions1 = list(shape_fuctions)

    shape_natural = []

    for i in range(len(shape_fuctions1)):
        temp = shape_fuctions1[i].subs(x, xx)
        temp1 = temp.subs(y, yy)
        temp2 = temp1.subs(z, zz)
        shape_natural.append(temp2)
    print(np.array(shape_natural))
    print(shape_natural[1].args)


def shape_functions_natural():
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

    N = [-0.5 - 0.5 * ksi - 0.5 * eta - 0.5 * dzeta,
         0.5 + 0.5 * ksi,
         0.5 + 0.5 * eta,
         0.5 + 0.5 * dzeta]

    return N

#nie uzywane/używane w jacobian a jacobian używany w assembling :)
def coordinate_change_model(vertices, natural_shape_fc):

    x = vertices[0, 0]*natural_shape_fc[0] + vertices[1, 0]*natural_shape_fc[1] +\
        vertices[2, 0]*natural_shape_fc[2] + vertices[3, 0]*natural_shape_fc[3]
    y = vertices[0, 1]*natural_shape_fc[0] + vertices[1, 1]*natural_shape_fc[1] +\
        vertices[2, 1]*natural_shape_fc[2] + vertices[3, 1]*natural_shape_fc[3]
    z = vertices[0, 2]*natural_shape_fc[0] + vertices[1, 2]*natural_shape_fc[1] +\
        vertices[2, 2]*natural_shape_fc[2] + vertices[3, 2]*natural_shape_fc[3]

    return x, y, z


def jacobian(vertices, natural_shape_fc):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')
    x_in_nat, y_in_nat, z_in_nat = coordinate_change_model(vertices, natural_shape_fc)
    j = sp.Matrix([[sp.diff(x_in_nat, ksi), sp.diff(y_in_nat, ksi), sp.diff(z_in_nat, ksi)],
                    [sp.diff(x_in_nat, eta), sp.diff(y_in_nat, eta), sp.diff(z_in_nat, eta)],
                   [sp.diff(x_in_nat, dzeta), sp.diff(y_in_nat, dzeta), sp.diff(z_in_nat, dzeta)]])

    return sp.det(j) # jacobian


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


