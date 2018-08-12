from MES_dir import config
from MES_dir import readData as rd
import numpy as np
import matplotlib.pyplot as plt


def comparison(number_of_curves_to_draw=20):
    lowF = -5
    highF = 75
    lowK = -5
    highK = 150
    lowV = -5
    highV = 10000

    path_to_kvect_file1 = config.ROOT_DIR + '/../eig/kvect1'
    path_to_kvect_file2 = config.ROOT_DIR + '/../eig/kvect2'
    path_to_kvect_file3 = config.ROOT_DIR + '/../eig/kvect3'
    path_to_omega_files1 = config.ROOT_DIR + '/../eig/omega1'
    path_to_omega_files2 = config.ROOT_DIR + '/../eig/omega2'
    path_to_omega_files3 = config.ROOT_DIR + '/../eig/omega3'

    kvect1 = np.array(rd.read_kvect(path_to_kvect_file1))
    kvect2 = np.array(rd.read_kvect(path_to_kvect_file2))
    kvect3 = np.array(rd.read_kvect(path_to_kvect_file3))
    k_v1 = kvect1 * 1e3
    k_v2 = kvect2 * 1e3
    k_v3 = kvect3 * 1e3

    curves = [i for i in range(number_of_curves_to_draw)]

    plt.figure(1)
    plt.subplot(211)

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files1, ind) / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v1, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([lowF, highF])#600
    plt.ylim([-5, highK])#2000

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files2, ind) / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v2, 'b.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([lowF, highF])#600
    plt.ylim([-5, highK])#2000

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files3, ind) / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v3, 'r.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([lowF, highF])#600
    plt.ylim([-5, highK])#2000

    plt.subplot(212)
    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files1, ind) / (2 * np.pi)
        v_p = (f_v / k_v1) * 2 * np.pi
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Phase velocity [m/s]")
    plt.xlim([lowF, highF])#500
    plt.ylim([-5, 10000])#50

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files2, ind) / (2 * np.pi)
        v_p = (f_v / k_v2) * 2 * np.pi
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v * 1e-3, v_p, 'b.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Phase velocity [m/s]")
    plt.xlim([lowF, highF])#500
    plt.ylim([-5, 10000])#50

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files3, ind) / (2 * np.pi)
        v_p = (f_v / k_v3) * 2 * np.pi
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v * 1e-3, v_p, 'r.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Phase velocity [m/s]")
    plt.xlim([lowF, highF])#500
    plt.ylim([-5, 10000])#50

    plt.show()

comparison()