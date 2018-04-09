import sympy as sp
import numpy as np
import scipy.linalg as la
from dispersion_curves import  functions
from MES_dir import config, gauss
import scipy.integrate as integr

x, y = sp.symbols('x, y')
ksi, eta = sp.symbols('ksi, eta')

def p_vector():
    x, y = sp.symbols('x, y')
    p = [1, x, y]
    return sp.Matrix(p).reshape(1, 3)


def me_inv_matrix(vertices, element_indices):
    me = []
    for ind in element_indices:
        temp = [1, vertices[ind, 0], vertices[ind, 1]]
        me.append(temp)
    return sp.Matrix(me).inv()


def shape_functions(vertices, element_indices):
    p = p_vector()
    me_inv = me_inv_matrix(vertices, element_indices)
    return p * me_inv  # N


def b_matrix_fc(shape_functions):
    b = []
    w1 = []
    w2 = []
    w3 = []

    for n in shape_functions:
        w1.append(sp.diff(n, x))
        w1.append(0)
        w1.append(0)

        w2.append(0)
        w2.append(sp.diff(n, y))
        w2.append(0)

        w3.append(sp.diff(n, y))
        w3.append(sp.diff(n, x))
        w3.append(0)

    b.append(w1)
    b.append(w2)
    b.append(w3)


    return np.array(b)


def d_matrix_fc(young_modulus, poisson_coeficient):
    a = young_modulus / ((1 + poisson_coeficient) * (1 - 2*poisson_coeficient))
    b1 = (1 - poisson_coeficient) * a
    b2 = (poisson_coeficient) * a
    b3 = ((1 - 2*poisson_coeficient) / 2) * a
    matrix = [[b1, b2, 0], [b2, b1, 0], [0, 0, b3]]
    return np.array(matrix)


def volume_det(vertices):
    #to samo wyznacznikiem
    temp1 = [1]
    temp1.extend(list(vertices[0]))
    temp2 = [1]
    temp2.extend(list(vertices[1]))
    temp3 = [1]
    temp3.extend(list(vertices[2]))

    a = [temp1, temp2, temp3]
    a1 = np.array(a)

    #dzielone przez 6 zeby wyszlo jak w drugiej funkcji - czy dobrze???
    return abs(la.det(a1)/2)


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


def shape_functions_natural():

    N = [- 0.5 * ksi - 0.5 * eta,
         0.5 + 0.5 * ksi,
         0.5 + 0.5 * eta]

    return N


def matrix_to_integrate(density):
    shape_functions = shape_functions_natural()

    N = np.array([[shape_functions[0], 0, 0, shape_functions[1], 0, 0,
                   shape_functions[2], 0, 0],
                 [0, shape_functions[0], 0, 0, shape_functions[1], 0,
                  0, shape_functions[2], 0]])

    # Macierz podcalkowa
    N_to_integrate = N.transpose().dot(N)*density

    return N_to_integrate


def coordinate_change_model(vertices, natural_shape_fc):


    x = vertices[0, 0]*natural_shape_fc[0] + vertices[1, 0]*natural_shape_fc[1] + vertices[2, 0]*natural_shape_fc[2]
    y = vertices[0, 1]*natural_shape_fc[0] + vertices[1, 1]*natural_shape_fc[1] + vertices[2, 1]*natural_shape_fc[2]

    return x, y


def jacobian(vertices, natural_shape_fc):
    x_ksi_eta, y_ksi_eta = coordinate_change_model(vertices, natural_shape_fc)
    j = sp.Matrix([[sp.diff(x_ksi_eta, ksi), sp.diff(y_ksi_eta, ksi)],
                    [sp.diff(x_ksi_eta, eta), sp.diff(y_ksi_eta, eta)]])

    return sp.det(j) # jacobian


def mass_local_matrix(density, vertices, natural_sf):

    j = jacobian(vertices, natural_sf)

    N_integrate = matrix_to_integrate(density)
    # integral limits
    eta_bot = lambda ksi: -1
    eta_top = lambda ksi: -ksi

    integral = []

    for row in N_integrate:
        integr_row = []
        for elem in row:
            if elem != 0:
                lam_elem = sp.lambdify((ksi, eta), elem*j)

                temp = integr.dblquad(lam_elem, -1, 1, eta_bot, eta_top)[0]

            if elem == 0:
                temp = 0

            integr_row.append(temp)
        integral.append(integr_row)
    return np.array(integral)



k = functions.read_MARC_matrix('trojkat_lokalny/model8_job1_glstif_0001.bin', 3, 3)
k1 = functions.read_MARC_matrix('trojkat_obrocony/model8_job1_glstif_0001.bin', 3, 3)
m = functions.read_MARC_matrix('trojkat_lokalny/model8_job1_glmass_0001.bin', 3, 3)


vert = functions.read_coordinates_from_DAT('trojkat_lokalny/model8_job1.dat', 3, 3)
vert1 = functions.read_coordinates_from_DAT('trojkat_obrocony/model8_job1.dat', 3, 3)

v = vert[0:4, 1:3]
v1 = vert1[0:4, 1:3]

ind = [0, 1, 2]

sf = shape_functions(v, ind)
d_matrix = d_matrix_fc(config.young_mod, config.poisson_coef,)
stiff = stiff_local_matrix(sf, v, ind, config.young_mod, config.poisson_coef)
mass = mass_local_matrix(config.density, v, shape_functions_natural())

sf1 = shape_functions(v1, ind)
d_matrix1 = d_matrix_fc(config.young_mod, config.poisson_coef,)
stiff1 = stiff_local_matrix(sf1, v1, ind, config.young_mod, config.poisson_coef)
mass1 = mass_local_matrix(config.density, v1, shape_functions_natural())
print("koniec")