import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from MES_dir import MES, config
from MES_dir import readData as rd
import time
from GUI import my_threads as threads


# Zwraca macierze K0, KL, KR, M0 do policzenia wartosci wlasnych ukladu
def getDataForEiq():
    dim = int(np.shape(config.k)[0]/3)
    print(dim)
    config.kl = config.k[dim:2 * dim, 0:dim]
    config.k0 = config.k[dim:2 * dim, dim:2 * dim]
    config.kr = config.k[dim:2 * dim, 2 * dim:3 * dim]
    config.ml = config.m[dim:2 * dim, 0:dim]
    config.m0 = config.m[dim:2 * dim, dim:2 * dim]
    config.mr = config.m[dim:2 * dim, 2 * dim:3 * dim]
    # config.ml = config.m_focused_rows[dim:2 * dim, 0:dim]
    # config.m0 = config.m_focused_rows[dim:2 * dim, dim:2 * dim]
    # config.mr = config.m_focused_rows[dim:2 * dim, 2 * dim:3 * dim]
    # writing to file
    rd.write_matrix_to_file('m_focused', config.m_focused_rows)
    rd.write_matrix_to_file("k", config.k)
    rd.write_matrix_to_file("m", config.m)
    rd.write_matrix_to_file("k0", config.k0)
    rd.write_matrix_to_file("kr", config.kr)
    rd.write_matrix_to_file("kl", config.kl)
    rd.write_matrix_to_file("ml", config.ml)
    rd.write_matrix_to_file("m0", config.m0)
    rd.write_matrix_to_file("mr", config.mr)

# Znajduje wektor wartosci wlasnych dla systemu
def findEig(saveEigVectors=False):
    getDataForEiq()

    ux = 1
    # kvect = np.linspace(1e-10, np.pi/8, num=51)
    kvect = np.linspace(config.kvect_min, config.kvect_max, num=config.kvect_no_of_points)
    path_to_kvect = config.ROOT_DIR + "/../eig/kvect"

    rd.write_vector_to_file(path_to_kvect, kvect)

    fsys = []

    for ind, k in enumerate(kvect):
        ksys = config.k0 + np.exp(-1j * ux * k)*config.kl + np.exp(+1j * ux * k)*config.kr
        msys = config.m0 + np.exp(-1j * ux * k)*config.ml + np.exp(+1j * ux * k)*config.mr
        # msys = config.m0

        [f, v] = la.eig(ksys, msys)

        f1 = np.sqrt(f)

        # zapisywanie do plikow eig
        if saveEigVectors:
            eig_list = []
            for i in range(len(f1)):

                eig_list.append(f1[i])
                temp = [v[i, j] for j in range(np.shape(v)[0])]
                eig_list.append(temp)
                rd.write_vector_to_file(config.ROOT_DIR + '/../eig/eig_{}'.format(k), eig_list)

        fsys.append(f1)
        print("eig ", ind+1, " z {}".format(config.kvect_no_of_points))
        print(np.shape(f1))
    return np.array(fsys).transpose(), kvect

def drawDispercionCurves(number_of_curves_to_draw=10, save_plot_to_file=False):
    start = time.clock()
    print("eig")
    fsys, kvect = findEig()
    print("po eig")
    plt.figure(1)
    plt.subplot(211)
    k_v = kvect * 1e3

    # for ind in range(len(fsys)):
    #     f_v = fsys[ind, :] / (2 * np.pi)
    #     plt.plot(f_v * 1e-3, k_v, 'g.', markersize=3)
    # plt.xlabel("Frequency [kHz]")
    # plt.ylabel("Wavenumber [rad/m]")
    # plt.xlim([0, 2000])#600
    # plt.ylim([0, 5000])#2000
    #
    # plt.subplot(212)
    # for ind in range(len(fsys)):
    #     f_v = fsys[ind, :] / (2 * np.pi)
    #     v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
    #     plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    # plt.xlabel("Frequency [kHz]")
    # plt.ylabel("Velocity [m/s]")
    # plt.xlim([0, 2000])#500
    # plt.ylim([0, 500])#50
    # plt.show()

    print("Obliczono wartosci wlasne. Czas: ", time.clock() - start, "[s]")
    print("Obliczono wartosci wlasne. Czas: ", (time.clock() - start)/60, "[min]")
    print("Obliczono wartosci wlasne. Czas: ", (time.clock() - start)/3600, "[h]")

    new_fsys = sortColumns(fsys)

    curves = [i for i in range(number_of_curves_to_draw)]
    # curves = [10]
    omega_path = config.ROOT_DIR + '/../eig/omega'
    rd.write_matrix_to_file(omega_path, new_fsys, length=50)

    for ind in curves:
        f_v = new_fsys[ind, :] / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([-5, 100])#600
    plt.ylim([-5, 200])#2000

    plt.subplot(212)
    for ind in curves:
        f_v = new_fsys[ind, :] / (2 * np.pi)
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        v_p = (f_v / k_v)* 2 * np.pi
        plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Velocity [m/s]")
    plt.xlim([-5, 100])#500
    plt.ylim([-5, 10000])#50

    if save_plot_to_file:
        plt.savefig('dis_curves.png')
    plt.show()

#Rysuje z zapisanych w folderze eig wartości własnych, bez obliczania ich.
def drawDispercionCurvesFromFile(number_of_curves_to_draw=30, save_plot_to_file=False):

    plt.figure(1)
    plt.subplot(211)

    path_to_kvect_file = config.ROOT_DIR + '/../eig/kvect'
    path_to_omega_files = config.ROOT_DIR + '/../eig/omega'
    kvect = np.array(rd.read_kvect(path_to_kvect_file))
    k_v = kvect * 1e3

    curves = [i for i in range(number_of_curves_to_draw)]
    # curves = [10]

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files, ind) / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([-5, 100])#600
    plt.ylim([-5, 200])#2000

    plt.subplot(212)
    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files, ind) / (2 * np.pi)
        v_p = (f_v / k_v) * 2 * np.pi
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Phase velocity [m/s]")
    plt.xlim([-5, 100])#500
    plt.ylim([-5, 10000])#50

    if save_plot_to_file:
        plt.savefig('dis_curves.png')
    plt.show()

def sortColumns(matrix):
    new_matrix = []
    print("wymair macierzy: ", np.shape(matrix))
    for i in range(len(matrix[0, :])):
        column = matrix[:, i]
        new_matrix.append(np.sort_complex(column))
    return np.array(new_matrix).transpose()




