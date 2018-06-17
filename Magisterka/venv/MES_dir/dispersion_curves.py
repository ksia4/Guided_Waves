import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from MES_dir import MES, config
from MES_dir import readData as rd
import time
from GUI import my_threads as threads


# Zwraca macierze K0, KL, KR, M0 do policzenia wartosci wlasnych ukladu
def get_data_for_eiq():
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
def find_eig(path='eig', other=False):
    get_data_for_eiq()

    ux = 1
    kvect = np.linspace(1e-10, np.pi/8, num=51)
    #kvect = np.linspace(config.kvect_min, config.kvect_max, num=config.kvect_no_of_points)
    path_to_kvect = path + '/kvect'

    rd.write_vector_to_file(path_to_kvect, kvect)

    fsys = []

    for ind, k in enumerate(kvect):
        ksys = config.k0 + np.exp(-1j * ux * k)*config.kl + np.exp(+1j * ux * k)*config.kr
        msys = config.m0 + np.exp(-1j * ux * k)*config.ml + np.exp(+1j * ux * k)*config.mr
        # msys = config.m0
        [f, v] = la.eig(ksys, msys)

        f1 = np.sqrt(f)

        # zapisywanie do plikow eig
        eig_list = []
        for i in range(len(f1)):
            eig_list.append(f1[i])
            temp = [v[i, j] for j in range(np.shape(v)[0])]
            eig_list.append(temp)
        if other:
            rd.write_vector_to_file('../eig/eig_{}'.format(k), eig_list)
        else:
            rd.write_vector_to_file('eig/eig_{}'.format(k), eig_list)


        fsys.append(f1)
        print("eig ", ind+1, " z {}".format(config.kvect_no_of_points))
        print(np.shape(f1))
    return np.array(fsys).transpose(), kvect

def draw_dispercion_curves(path_to_folder_with_data='eig', save_plot_to_file=False):
    start = time.clock()

    fsys, kvect = find_eig(path_to_folder_with_data, True)

    plt.figure(1)
    plt.subplot(311)
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

    new_fsys = sort_columns(fsys)

    curves = [i for i in range(10)]
    # curves = [10]
    omega_path = path_to_folder_with_data + '/omega'
    rd.write_matrix_to_file(omega_path, new_fsys, length=50)

    for ind in curves:
        f_v = new_fsys[ind, :] / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([-5, 200])#600
    plt.ylim([-5, 400])#2000

    plt.subplot(312)
    for ind in curves:
        f_v = new_fsys[ind, :] / (2 * np.pi)
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        v_p = f_v / k_v
        plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Velocity [m/s]")
    plt.xlim([-5, 200])#500
    plt.ylim([-5, 5000])#50

    plt.subplot(313)
    for ind in curves:
        f_v = new_fsys[ind, :] / (2 * np.pi)
        v_g = calculate_group_velocity(f_v, kvect)
        plt.plot(f_v[0: -1] * 1e-3, v_g, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Velocity [m/s]")
    plt.xlim([-5, 200])#500
    plt.ylim([-5, 700])#50
    if save_plot_to_file:
        plt.savefig('dis_curves.png')
    plt.show()

#Rysuje z zapisanych w folderze eig wartości własnych, bez obliczania ich.
def draw_dispercion_curves_from_file(path_to_folder_with_data='eig', number_of_curves_to_draw=10, save_plot_to_file=False):

    plt.figure(1)
    plt.subplot(311)

    path_to_kvect_file = path_to_folder_with_data + '/kvect'
    path_to_omega_files = path_to_folder_with_data + '/omega'
    kvect = np.array(rd.read_kvect(path_to_kvect_file))
    k_v = kvect * 1e3

    curves = [i for i in range(number_of_curves_to_draw)]
    # curves = [10]

    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files, ind) / (2 * np.pi)
        plt.plot(f_v * 1e-3, k_v, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([-5, 200])#600
    plt.ylim([-5, 400])#2000

    plt.subplot(312)
    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files, ind) / (2 * np.pi)
        v_p = f_v / k_v
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v * 1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Phase velocity [m/s]")
    plt.xlim([-5, 200])#500
    plt.ylim([-5, 5000])#50

    plt.subplot(313)
    for ind in curves:
        f_v = rd.read_complex_omega(path_to_omega_files, ind) / (2 * np.pi)
        v_g = calculate_group_velocity(f_v, kvect)
        # v_p = (2 * np.pi * (f_v / k_v)) / (np.sqrt(config.young_mod / config.density) * 1e-3)
        plt.plot(f_v[0: -1] * 1e-3, v_g, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Group velocity [m/s]")
    plt.xlim([-5, 200])
    plt.ylim([-5, 700])
    if save_plot_to_file:
        plt.savefig('dis_curves.png')
    plt.show()

def sort_columns(matrix):
    new_matrix = []
    print("wymair macierzy: ", np.shape(matrix))
    for i in range(len(matrix[0, :])):
        column = matrix[:, i]
        new_matrix.append(np.sort_complex(column))
    return np.array(new_matrix).transpose()

def calculate_group_velocity(f_for_mode, kvect):
    group_vel = []
    for ind in range(len(kvect)-1):
        numerator = f_for_mode[ind + 1] - f_for_mode[ind]
        denominator = kvect[ind + 1] - kvect[ind]
        denominator *= 1e3
        group_vel.append(numerator.real/denominator)
    return np.array(group_vel)

