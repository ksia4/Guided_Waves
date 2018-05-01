from MES_dir import mesh, selectMode, config
from MES_dir import dispersion as disp
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

def calculate_n(k_Nyquista, delta_k, factor=1.1):

    # Funkcja obliczająca porządaną ilość próbek w wektorze odległości w końcowym śladzie
    # powinien być zbliżony do długości wektora czasu
    # n > 2(k_Nyquista/deltak)
    if factor <= 1:
        print("Współczynnik musi być większy od 1, przyjęta wartość wynosi 1,1")
        factor = 1.1

    return int(factor * 2 * (k_Nyquista/delta_k))

# def calculate_n2(delta_x, delta_k, k_Nyquista):
#     #Obliczam n z równania n=1/(delta_k*delta_x)
#     n = 1/(delta_k*delta_x)
#     #sprawdzam warunek n > 2*(k_Nyquista/delta_k)
#     if n > 2*(k_Nyquista/delta_k):
#         print("Wszystko ok")
#     else:
#         print("Zupa z trupa")
#
#     return n

def calculate_k_nyquist(dispercion_curves, dt, factor=1.1):
    #k_Nyquista powinno być >= k(f_Nyquista) f_Nyquista to 1/(2*delta_t)
    #Zależność k(f) przechowywana jest w krzywych dyspersji

    if factor <= 1:
        print("Podany współczynnik musi być większy od 1, przyjęto wartość 1,1")
        factor = 1.1
    f_Nyq = 1/(2*dt) # to jest w Hz
    f_Nyq_kHz = f_Nyq/1000
    max_k_Nyq = 0
    for mode in dispercion_curves.AllModes.modeTable:
        k_temp = Anim_dyspersji.curve_sampling(mode.all_omega_khz, dispercion_curves.k_v, [f_Nyq_kHz])
        if k_temp > max_k_Nyq:
            max_k_Nyq = k_temp

    return factor*max_k_Nyq[0] # Zwracana wartość jest w rad/m

def calculate_delta_k(dispercion_curves, signal_duration, factor=0.9):
    # delta k powinno być = 1/(n(delta_x) i mniejsze niż 1/(m*delta_t*v_gr_max) m*delta_t jest równe długości trwania sygnału :)
    if signal_duration <= 0:
        print("Długość sygnału musi być większa od 0")
        exit(0)
    if factor >= 1:
        print("Współczynnik musi być mniejszy od 1, przyjęta wartość to 0,9")
        factor = 0.9
    max_v_gr = 0
    for mode in dispercion_curves.AllModes.modeTable:
        temp_v_gr = 1000 * max(disp.calculate_group_velocity(mode.all_omega_khz, dispercion_curves.k_v/1000))
        if temp_v_gr > max_v_gr:
            max_v_gr = temp_v_gr
    # print("Prędkość grupowa max = " + str(max_v_gr))
    delta_k = factor/(signal_duration * max_v_gr)
    return delta_k # delta_k zwracana jest w rad/m

def calculate_delta_x(k_Nyquista):
    return 1/(2*k_Nyquista) # w metrach

def check_all_conditions(n, delta_k, delta_x, k_nyq):
    if(n < 2*(k_nyq/delta_k)):
        print("nierówność na n jest niespełniona")
    if delta_k != 1/(n*delta_x):
        print("Warunek na delte k nie jest spełniony, delta_k wynosi:")
        print(delta_k)
        print("natomiast 1/(n*deltax)")
        print(1/(n*delta_x))
    if k_nyq != 1/(2*delta_x):
        print("Warunek na k_Nyquista nie jest spełniony")

def find_max_k(mode, k_vect, max_omega_kHz):
    print(max_omega_kHz)
    print(mode.all_omega_khz[-1])
    print(mode.all_omega_khz[-2])
    print(k_vect[-1])
    print(k_vect[-2])
    if max_omega_kHz > mode.all_omega_khz[-1]:
        max_k = mode.findPoint([mode.points[-2], mode.points[-1]], max_omega_kHz)
        print(max_k)
    elif max_omega_kHz < mode.minOmega:
        print("Ten mod nie zostanie wzbudzony")
        max_k = -1
    else:
        P1 = selectMode.Point()
        P2 = selectMode.Point()
        for ind in range(len(mode.points)-1):
            if mode.points[ind].w < max_omega_kHz and mode.points[ind+1].w > max_omega_kHz:
                P1 = mode.points[ind]
                P2 = mode.points[ind+1]
                break
        max_k = mode.findPoint([P1, P2], max_omega_kHz)
        print(max_k)
    return max_k

def find_omega_in_dispercion_curves(mode, temp_k, k_vect):
    omega = mode.points[0].w
    if temp_k > k_vect[-1]:
        omega = mode.findPointWithGivenK([mode.points[-2], mode.points[-1]], temp_k)
    elif temp_k < k_vect[0]:
        if mode.points[0].w < 5:
            temp_point = selectMode.Point()
            omega = mode.findPointWithGivenK([temp_point, mode.points[0]], temp_k)
        else:
            omega = mode.points[0].w
    else:
        for ind in range(len(k_vect)-1):
            if k_vect[ind] < temp_k and k_vect[ind + 1] > temp_k:
                omega = mode.findPointWithGivenK([mode.points[ind], mode.points[ind+1]], temp_k)
    return omega

def find_omega_in_dispercion_curves_rad_s(mode, temp_k, k_vect):
    omega = mode.points[0].wkat_complex
    if temp_k > k_vect[-1]:
        omega = mode.findPointWithGivenK_rad_s([mode.points[-2], mode.points[-1]], temp_k)
    elif temp_k < k_vect[0]:
        if mode.points[0].w < 5:
            temp_point = selectMode.Point()
            omega = mode.findPointWithGivenK_rad_s([temp_point, mode.points[0]], temp_k)
        else:
            omega = mode.points[0].wkat_complex
    else:
        for ind in range(len(k_vect)-1):
            if k_vect[ind] < temp_k and k_vect[ind + 1] > temp_k:
                omega = mode.findPointWithGivenK_rad_s([mode.points[ind], mode.points[ind+1]], temp_k)
                break
    return omega

def find_value_by_omega_in_G_w(G_w, freq_sampling_kHz, omega):
    value = -1
    for ind in range(len(freq_sampling_kHz)-1):
        if freq_sampling_kHz[ind] == omega:
            value = G_w[ind]
            break
        elif freq_sampling_kHz[ind] < omega and freq_sampling_kHz[ind + 1] > omega:
            a = (G_w[ind] - G_w[ind+1])/(freq_sampling_kHz[ind] - freq_sampling_kHz[ind +1])
            b = G_w[ind] - a * freq_sampling_kHz[ind]
            value = a* omega + b
            break
    if value == -1:
        if omega == freq_sampling_kHz[-1]:
            value = G_w[-1]
    return value

def calculate_group_velocity(mode, k_sampling_rad_m, ind, k_vect):
    k1 = k_sampling_rad_m[ind + 1]
    k2 = k_sampling_rad_m[ind]
    om1 = find_omega_in_dispercion_curves_rad_s(mode, k1, k_vect)
    om2 = find_omega_in_dispercion_curves_rad_s(mode, k2, k_vect)

    group_velocity = (om1 - om2)/(k1 - k2)
    return group_velocity

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
    dist = 4 # w metrach
    # KrzyweDyspersji.plot_modes(50)

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()
    # for i in range(length):
    print("Zaraz będzie się dzało :o")
    # make_dispersion_in_bar(length, len(plane), dx, KrzyweDyspersji)
    dispersion = Anim_dyspersji.draw_time_propagation(signal_array, time_x_freq, dist, KrzyweDyspersji)
    plt.plot(time_x_freq[0], signal_array[3])
    plt.show()
    plt.plot(dispersion[0], dispersion[1])
    plt.show()

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()

    # signal_to_fft = pad_timetraces_zeroes(dispersion[0], dispersion[1])
    signal_to_fft = dispersion
    signal_after_fft = np.fft.rfft(signal_to_fft[1])
    freq_sampling = time_x_freq[3]
    time = time_x_freq[0]
    dt = time[-1]/len(time)
    new_freq_sampling = np.linspace(freq_sampling[0], freq_sampling[-1], len(signal_after_fft))
    new_freq_sampling_kHz = new_freq_sampling*1e-3
    G_w = np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2)
    plt.plot(new_freq_sampling_kHz, G_w, '*')
    plt.show()


    k_nyq = calculate_k_nyquist(KrzyweDyspersji, dt)
    delta_x = calculate_delta_x(k_nyq)
    delta_k = calculate_delta_k(KrzyweDyspersji, time[-1])
    n = calculate_n(k_nyq, delta_k) # n to długość wektora x, liczba próbek na odległości

    mode_0 = KrzyweDyspersji.getMode(0)
    k_vect = KrzyweDyspersji.k_v
    max_k = find_max_k(mode_0, k_vect, new_freq_sampling_kHz[-1])
    new_k_sampling_rad_m = []
    k = 0
    print("Tworzenie wektora k")
    while k < max_k:
        new_k_sampling_rad_m.append(k)
        k += delta_k
    print("Utworzono wektor k ma on " + str(len(new_k_sampling_rad_m)) + "elementów")

    G_k = []
    ind = 0
    print("Teraz będziemy robić G(k)")
    for temp_k in new_k_sampling_rad_m:
        om = find_omega_in_dispercion_curves(mode_0, temp_k, k_vect)
        # val = find_value_by_omega_in_G_w(G_w, new_freq_sampling_kHz, om)
        val = find_value_by_omega_in_G_w(signal_after_fft, new_freq_sampling_kHz, om)
        G_k.append(val)
        print(ind)
        ind += 1


    plt.plot(new_freq_sampling_kHz, G_w)
    plt.show()
    plt.plot(new_k_sampling_rad_m, np.sqrt(np.array(G_k).real**2 + np.array(G_k).imag**2))
    plt.show()

    v_gr = []
    print("Teraz będę liczyć prędkość grupową")
    for ind in range(len(new_k_sampling_rad_m) - 1):
        print("Indeks wynosi " + str(ind))
        value = calculate_group_velocity(mode_0, new_k_sampling_rad_m, ind, k_vect)
        v_gr.append(value)

    v_gr.append(v_gr[-1])
    plt.plot(new_k_sampling_rad_m, v_gr)
    plt.show()

    H_k = []
    for ind in range(len(v_gr)):
        H_k.append(G_k[ind] * v_gr[ind])

    plt.plot(new_k_sampling_rad_m, np.sqrt(np.array(H_k).real**2 + np.array(H_k).imag**2))
    plt.show()
    print("długość H(k) to " + str(len(H_k)))

    h_x = np.fft.ifft(H_k)
    distance = 1/delta_k # w metrach
    n = len(h_x)
    dx = distance/n
    print("dx wynosi " + str(dx) + "A cały dystans " + str(distance))
    dist_vect = []
    for i in range(n):
        dist_vect.append(i*dx*6)

    plt.plot(dist_vect, h_x)
    plt.show()

    # print(k_vect[-1]/delta_k)
    # print(len(signal_after_fft.real))
    # if k_vect[-1]/delta_k < len(signal_after_fft.real) - 1:
    #     delta_k = k_vect[-1]/(len(signal_after_fft.real)-1)
    # print(delta_k)
    # new_k_sampling = []
    # k = 0
    # while k < k_vect[-1]:
    #     new_k_sampling.append(k)
    #     # omega = find_omega_with_k(k, k_vect, mode_0)
    #     # omega = Anim_dyspersji.curve_sampling(mode_0.all_omega_khz, k_vect, [k])
    #     k += delta_k
    # new_omega_vector = Anim_dyspersji.curve_sampling(k_vect, mode_0.all_omega_khz, new_k_sampling)
    # G_k = Anim_dyspersji.curve_sampling(mode_0.all_omega_khz, np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2), new_omega_vector)
    # plt.plot(new_omega_vector)
    # plt.show()
    # k = 0
    # v_gr = []
    # for k in range(len(G_k)-1):
    #     v_gr.append((new_omega_vector[k+1]-new_omega_vector[k])/delta_k)
    #
    # H_k = []
    #
    # for ind in range(len(v_gr)):
    #     H_k.append(G_k[ind]*v_gr[ind])
    #
    # new_signal = np.fft.irfft(H_k)
    # print(len(new_signal))
    # print(n*delta_x)
    # plt.plot(new_signal)
    # plt.show()
    # check_all_conditions(n, delta_k, delta_x, k_nyq) Nie ma sensu tego sprawdzać...

    # chirp, time_x_frq = make_chirp(0, 1e5, 1e-4, True)
    # timeTraces = make_disp(chirp, time_x_frq[0], time_x_frq[1], time_x_frq[2], length, dx, KrzyweDyspersji)
    # animDisp(vertices, len(plane), length)

    #
# signal_array, time_x_freq = chirp_propagation.get_chirp()
