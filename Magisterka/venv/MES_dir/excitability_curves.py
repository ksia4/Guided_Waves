import numpy as np
import numpy.linalg as la
from MES_dir import config
from MES_dir import readData as rd
import matplotlib.pyplot as plt
from Propagation import selectMode


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
        print(om)
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

    plt.plot(new_omega, abs_a)

    return a

def calculate_and_show_curves(numberOfModes):
    # wczytywanie macierzy k, m, itd..
    print("Wczytywanie macierzy mas i sztywności z plików")
    rd.read_matricies()
    #zakladanie sily
    print("Wprowadzanie wymuszenia")
    force1 = np.zeros(np.shape(config.k)[0])
    force1[0] = 10
    config.force = np.array(force1)

    for mode in range(numberOfModes):
        print("Obliczenia dla modu ", mode)
        calculate_excitablity(mode, config.force)
    plt.legend
    plt.show()


def draw_curves_from_files(arguments):
    # kvect = rd.read_kvect("../eig/kvect")

    plt.figure(1)

    for ind in range(3):
        path = "../interpolated_curves/mode_{}".format(ind)
        polynomial = rd.read_interpolated_fc(path)
        values = calculate_function_values(arguments, polynomial)
        plt.plot(arguments, values)

    plt.show()

calculate_and_show_curves(4)

#------- Wersja z posortowanymi modami ---------
# KrzyweDyspersji=selectMode.SelectedMode('../eig/kvect', '../eig/omega')
# KrzyweDyspersji.selectMode()
# calculate_and_show_curves2(KrzyweDyspersji, 10)
