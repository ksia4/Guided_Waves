import numpy as np
import numpy.linalg as la
from MES_dir import config
from MES_dir import readData as rd
import matplotlib.pyplot as plt
from MES_dir import normalization


# Dla wektora lub 2-wymiarowej tablicy
def hermitian_transpose(matrix):

    final_matrix = []

    if matrix.ndim == 2:
        matrix1 = np.transpose(matrix)
        for row in matrix1:
            temp = []
            for element in row:
                temp.append(np.conjugate(element))
            final_matrix.append(temp)
        return np.array(final_matrix)

    if matrix.ndim == 1:
        for element in matrix:
            final_matrix.append(np.conjugate(element))
        return np.array(final_matrix)


# oblicza P do wzoru na wzbudzalnosc
def calculate_p(kr, kl, wavenumber, eigvector):
    # obliczanie sprzezen hermitowskich
    eigenvector_h = hermitian_transpose(eigvector)
    kr_h = hermitian_transpose(kr)
    kl_h = hermitian_transpose(kl)

    ux = 1
    temp = kr_h*np.exp(-1j*wavenumber*ux) - kl_h*np.exp(1j*wavenumber*ux) -\
            kr*np.exp(1j*wavenumber*ux) + kl*np.exp(-1j*wavenumber*ux)

    p1 = np.dot(eigenvector_h, temp)
    p = np.dot(p1, eigvector)
    return p


def calculate_displacement(omega):
    m = config.m * (omega * 1e3 ) ** 2
    temp = config.k - m
    x = np.dot(la.inv(temp), config.force)
    return np.transpose(x)


def calculate_excitablity(mode, f):
    # k_vect = np.linspace(1e-10, np.pi / 8, num=51)
    k_vect = rd.read_kvect("../eig/kvect")
    omega = rd.read_complex_omega("../eig/omega", mode)
    omega_string = rd.read_string_omega("../eig/omega", mode)
    a = []
    for kv, om in zip(k_vect, omega_string):
        # x = calculate_displacement(om)
        # dim = int(len(x)/3)
        # x0 = x[dim : 2 * dim]

        # eigv = rd.readEigMap("../eig/normeig/normeig_{}".format(kv), str(om))

        eigv = rd.readEigMap("../eig/eig_{}".format(kv), eigval=om)

        p = calculate_p(config.kr, config.kl, kv, eigv)

        temp = np.dot(eigv, f[0:len(eigv)])
        a.append(temp/p)


    abs_a = []
    for aa in a:
        abs_a.append(abs(aa.imag))

    new_omega = []
    for om in omega:
        new_omega.append(om.real*1e-3/(2*np.pi))

    # print("Interpolacja")
    # polynomial = interpolation(new_omega, abs_a)
    # rd.write_vector_to_file("../interpolated_curves/mode_{}".format(mode), polynomial)

    plt.plot(new_omega, abs_a)

    return a


def calculate_and_show_curves():
    # wczytywanie macierzy k, m, itd..
    # normalization.createNormFile("../eig/kvect")
    print("Wczytywanie macierzy mas i sztywności z plików")
    rd.read_matricies()
    #zakladanie sily
    print("Wprowadzanie wymuszenia")
    force1 = np.zeros(np.shape(config.k)[0])
    force1[0] = 10
    config.force = np.array(force1)

    for mode in range(4):
        print("Obliczenia dla modu ", mode)
        calculate_excitablity(mode, config.force)
    plt.legend
    plt.show()


# DO PRZEROBIENIA WSZYSTKO PONIZEJ
def interpolation(arguments, values):
    #finding values for Ax=b equation, where b are values

    a_matrix = []
    for arg in arguments:
        temp = []
        a = 1
        for i in range(len(arguments)):
            temp.append(a)
            a *= arg
        a_matrix.append(temp)

    polynom = la.solve(a_matrix, values)
    return np.array(polynom)


def calculate_function_values(arguments, fc_coeffs):

    values = []
    for arg in arguments:
        val = 0
        a = 1
        for i in range(len(fc_coeffs)):
            val += a*fc_coeffs[i]
            a *= arg
        values.append(val)
    return np.array(values)


def draw_curves_from_files(arguments):
    # kvect = rd.read_kvect("../eig/kvect")

    plt.figure(1)

    for ind in range(3):
        path = "../interpolated_curves/mode_{}".format(ind)
        polynomial = rd.read_interpolated_fc(path)
        values = calculate_function_values(arguments, polynomial)
        plt.plot(arguments, values)

    plt.show()


# arguments = np.array([1000*i for i in range(200)])
# draw_curves_from_files(arguments)

calculate_and_show_curves()