from MES_dir import mesh, selectMode, config
from Animation import Anim_dyspersji
import matplotlib.pyplot as plt
import numpy as np


def find_accurate_len(actual_len):
    min_len = 8 * actual_len
    estimated_len = 256
    while estimated_len < min_len:
        estimated_len *= 2
    return estimated_len


def pad_timetraces_zeroes(time_vector, signal_vector):
    actual_len = len(signal_vector)
    print("teraz było tyle punktów")
    estimated_len = find_accurate_len(actual_len)
    print(actual_len)
    print("teraz będzie o tyle punktów")
    print(estimated_len)
    print("a czasopunktówy było o tyle")
    print(len(time_vector))
    total_multiple = int((estimated_len - actual_len)/(actual_len-1))
    print("Czyli po tyle w każdą przerwę")
    print(total_multiple)
    print("Plus na końcu jeszcze tyle próbek:")
    num_of_added_points = estimated_len - total_multiple*(actual_len-1) - actual_len
    print(num_of_added_points)
    temp_time_vector = np.linspace(time_vector[0], time_vector[-1], (actual_len-1)*total_multiple + len(time_vector)) # nowy wektor czasowy o takim samym przedziale czasowym, ale zagęszczony
    new_dt = temp_time_vector[1]-temp_time_vector[0]
    estimated_time_vector = []
    for i in range(estimated_len):
        if(i < len(temp_time_vector)):
            estimated_time_vector.append(temp_time_vector[i])
        else:
            estimated_time_vector.append(estimated_time_vector[-1] + new_dt)
    new_signal = []

    for signal_value in signal_vector:
        new_signal.append(signal_value)
        for zeros in range(total_multiple):
            new_signal.append(0)
    print("Przed dodaniem dodatkowych zer:")
    print(len(new_signal))
    for zeros in range(num_of_added_points-total_multiple):
        new_signal.append(0)

    print("time vector")
    print(len(estimated_time_vector))
    print("signal")
    print(len(new_signal))

    plt.plot(estimated_time_vector, new_signal)
    plt.show()

    return [estimated_time_vector, new_signal]




if __name__ == "__main__":
    # parametry preta
    bar_length = 1
    length = 100
    dx=bar_length/length
    radius = 10
    num_of_circles = 6
    num_of_points_at_c1 = 6

    # wektor liczby falowej
    config.kvect_min = 1e-10
    config.kvect_max = np.pi / 2
    config.kvect_no_of_points = 101

    # obliczenia
    # plane = mesh.circle_mesh_full(1, radius, num_of_circles, num_of_points_at_c1)
    vertices = mesh.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    # draw_bar(vertices, len(plane), length)
    KrzyweDyspersji=selectMode.SelectedMode('../eig/kvect', '../eig/omega')
    KrzyweDyspersji.selectMode()
    dist = 2 # w metrach
    # KrzyweDyspersji.plot_modes(50)

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()
    # for i in range(length):
    print("Zaraz będzie się dzało :o")
    # make_dispersion_in_bar(length, len(plane), dx, KrzyweDyspersji)
    dispersion = Anim_dyspersji.draw_time_propagation(signal_array, time_x_freq, dist, KrzyweDyspersji)

    plt.plot(dispersion[0], dispersion[1])
    plt.show()

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()

    signal_to_fft = pad_timetraces_zeroes(dispersion[0], dispersion[1])
    signal_after_fft = np.fft.rfft(signal_to_fft[1])
    freq_sampling = time_x_freq[3]
    print(len(freq_sampling))
    print("A długość fft to:")
    print(len(signal_after_fft))
    new_freq_sampling = np.linspace(freq_sampling[0], freq_sampling[-1], len(signal_after_fft))
    plt.plot(new_freq_sampling*1e-3, np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2))
    plt.show()

    # chirp, time_x_frq = make_chirp(0, 1e5, 1e-4, True)
    # timeTraces = make_disp(chirp, time_x_frq[0], time_x_frq[1], time_x_frq[2], length, dx, KrzyweDyspersji)
    # animDisp(vertices, len(plane), length)

    #
    # signal_array, time_x_freq = chirp_propagation.get_chirp()
