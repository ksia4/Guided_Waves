import numpy as np
import numpy.linalg as la
from MES_dir import config
from MES_dir import readData as rd
import matplotlib.pyplot as plt
from Propagation import selectMode


# Dla wektora lub 2-wymiarowej tablicy
def hermitianTranspose(matrix):

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
def calculateP(kr, kl, wavenumber, eigvector):
    # obliczanie sprzezen hermitowskich
    eigenvector_h = hermitianTranspose(eigvector)
    kr_h = hermitianTranspose(kr)
    kl_h = hermitianTranspose(kl)

    ux = 1
    temp = kr_h*np.exp(-1j*wavenumber*ux) - kl_h*np.exp(1j*wavenumber*ux) -\
            kr*np.exp(1j*wavenumber*ux) + kl*np.exp(-1j*wavenumber*ux)

    p1 = np.dot(eigenvector_h, temp)
    p = np.dot(p1, eigvector)
    return p


def calculateExcitablity(mode, f):
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

        p = calculateP(config.kr, config.kl, kv, eigv)

        temp = np.dot(eigv, f[0:len(eigv)])
        a.append(temp/p)

    abs_a = []
    for aa in a:
        abs_a.append(abs(aa.imag))

    frequency = []
    for om in omega:
        frequency.append(om.real*1e-3/(2*np.pi))

    return frequency, abs_a


def calculateAndShowCurves(numberOfModes):
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
        frequency, abs_a =calculateExcitablity(mode, config.force)
        plt.plot(frequency, abs_a)
    plt.legend
    plt.show()

