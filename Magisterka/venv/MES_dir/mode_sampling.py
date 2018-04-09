import numpy as np
import matplotlib.pyplot as plt
from MES_dir import readData as rd


def find_linear_fc(x1, x2, y1, y2):
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a * x1
    return np.array([b, a]) # od najmniejszej potegi zmiennej


def curve_sampling(omega, values, freq_sampled):
    k = [] # szukane k dla zadanych czestotliwosci -> frequencies

    for om in freq_sampled * (2 * np.pi):

        for i in range(np.shape(omega)[0] - 1):
            if om > omega[i].real and om < omega[i + 1].real:
                linera_fc = find_linear_fc(omega[i].real, omega[i + 1].real, values[i], values[i + 1])
                sampled_k = linera_fc[0] + om * linera_fc[1]
                # print("k: ", sampled_k)
                k.append(sampled_k)
                break
        if om <= omega[0]:
            k.append(values[0])
        if om >= omega[-1]:
            k.append(values[-1])
    return np.array(k)


def drawing_modes(number_of_modes):
    for mode in range(number_of_modes):
        path1 = "../eig/omega"
        path2 = "../eig/kvect"

        omega = rd.read_complex_omega(path1, mode)
        real_frequency = omega.real/(2 * np.pi)
        kvect = rd.read_kvect(path2)
        samples = 1000

        frequencies = np.linspace(min(real_frequency) + 1e-10, min(real_frequency) + 200e3, samples)


        k = mode_sampling(omega, kvect, frequencies)

        plt.plot(frequencies*1e-3, k*1e3)
    plt.xlim([-2, 200])#600
    plt.ylim([-2, 400])#2000
    plt.show()


# drawing_modes(1)
# omega = rd.read_complex_omega("../eig/omega",1)
# kvect = rd.read_kvect("../eig/kvect")
# freq_samples = np.linspace(0, 1e5 * 2, 10000)
# k = curve_sampling(omega, kvect, freq_samples)
#
# plt.plot(freq_samples, k)
# plt.show()
