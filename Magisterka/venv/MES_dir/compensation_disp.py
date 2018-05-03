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

def calculate_k_nyquist_in_function(dispercion_curves_of_propagated_mode, k_vector_to_dispercion_curves, dt, factor=1.1):
    #k_Nyquista powinno być >= k(f_Nyquista) f_Nyquista to 1/(2*delta_t)
    #Zależność k(f) przechowywana jest w krzywych dyspersji

    if factor <= 1:
        print("Podany współczynnik musi być większy od 1, przyjęto wartość 1,1")
        factor = 1.1
    f_Nyq = 1/(2*dt) # to jest w Hz
    f_Nyq_kHz = f_Nyq/1000
    max_k_Nyq = 0
    for mode in dispercion_curves_of_propagated_mode:
        k_temp = Anim_dyspersji.curve_sampling_new(mode.all_omega_khz, k_vector_to_dispercion_curves, [f_Nyq_kHz])
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

def calculate_delta_k_in_function(dispercion_curves, signal_duration, k_vector, factor=0.9):
    # delta k powinno być = 1/(n(delta_x) i mniejsze niż 1/(m*delta_t*v_gr_max) m*delta_t jest równe długości trwania sygnału :)
    if signal_duration <= 0:
        print("Długość sygnału musi być większa od 0")
        exit(0)
    if factor >= 1:
        print("Współczynnik musi być mniejszy od 1, przyjęta wartość to 0,9")
        factor = 0.9
    max_v_gr = 0
    for mode in dispercion_curves:
        # temp_v_gr = calculate_max_group_velocity_for_single_mode(mode, k_vector)
        temp_v_gr = 1000 * max(disp.calculate_group_velocity(mode.all_omega_khz, k_vector/1000))
        if temp_v_gr > max_v_gr:
            max_v_gr = temp_v_gr
    print("Prędkość grupowa max = " + str(max_v_gr))
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

def find_value_by_k_in_G_k(G_k, actual_k_vector, temp_k):
    value = -1
    for ind in range(len(actual_k_vector) - 1):
        if actual_k_vector[ind] == temp_k:
            value = G_k[ind]
            break
        elif actual_k_vector[0] > temp_k:
            a = G_k[0]/actual_k_vector[0]
            value = a * temp_k
        elif actual_k_vector[ind] < temp_k and actual_k_vector[ind + 1] > temp_k:
            a = (G_k[ind] - G_k[ind+1])/(actual_k_vector[ind] - actual_k_vector[ind +1])
            b = G_k[ind] - a * actual_k_vector[ind]
            value = a* temp_k + b
            break
    if value == -1:
        if temp_k == actual_k_vector[-1]:
            value = G_k[-1]
    return value

def calculate_max_group_velocity_for_single_mode(mode, k_vect):
    max_group_velocity = 0
    for ind in range(len(k_vect)-1):
        temp_group_velocity = (mode.allOmega[ind + 1].real - mode.allOmega[ind].real)/(k_vect[ind + 1] - k_vect[ind])
        if temp_group_velocity > max_group_velocity:
            max_group_velocity = temp_group_velocity
    return max_group_velocity

def calculate_group_velocity(mode, k_sampling_rad_m, ind, k_vect):
    k1 = k_sampling_rad_m[ind + 1]
    k2 = k_sampling_rad_m[ind]
    om1 = find_omega_in_dispercion_curves_rad_s(mode, k1, k_vect)
    om2 = find_omega_in_dispercion_curves_rad_s(mode, k2, k_vect)

    group_velocity = (om1 - om2)/(k1 - k2)
    return group_velocity

# def find_k_in_mode(modes_all_omega_khz, k_vector, omega):
#     if omega < modes_all_omega_khz[0]:
#         if modes_all_omega_khz < 5:



def calculate_mean_mode(dispercion_curves, numbers_of_propagated_modes):
    print("Robię średni mod")
    modes = []
    for ind in numbers_of_propagated_modes:
        modes.append(dispercion_curves.getMode(ind))

    mean_mode = selectMode.Mode()
    mean_k_vector = []
    omegs = modes[0].all_omega_khz
    for ind in range(len(omegs)):
        mean_k = modes[0].points[ind].k
        for mode_ind in range(len(modes)-1):
            # mean_k = mean_k + find_k_in_mode(modes[mode_ind + 1].all_omega_khz, dispercion_curves.k_v, omegs[ind])
            # print("z tego modu wyliczamy k")
            # plt.plot(modes[mode_ind + 1].all_omega_khz, dispercion_curves.k_v)
            # plt.show()
            calc_k = Anim_dyspersji.curve_sampling_new(modes[mode_ind + 1].all_omega_khz, dispercion_curves.k_v, [omegs[ind]])[0]
            # print("O takie mi wyszło to k " + str(calc_k))
            # print("A takie było w tym pierwszym modzie, powinny być podobne bardzo " + str(mean_k))
            mean_k = mean_k + calc_k
        mean_k = mean_k/(len(modes))
        mean_mode.addPoint(modes[0].points[ind])
        mean_mode.points[ind].k = mean_k
        mean_k_vector.append(mean_k)
    plt.plot(mean_mode.all_omega_khz, mean_k_vector)
    plt.show()
    return [mean_mode, mean_k_vector]

def transpose_G_w_to_G_k(mode, k_vector, frequency_sampling_kHz):
    transopsed_k_vector = Anim_dyspersji.curve_sampling_new(mode.all_omega_khz, k_vector, frequency_sampling_kHz)
    return transopsed_k_vector

def compensate_dispercion_Wilcox_method(received_signal_vector, time_vector, dispercion_curves_of_propagated_mode, k_vector_to_dispercion_curves):
    print("To wcale nei działa :(")


def gupi_Wilcox_Bedzie_dzialac(dispersion, dispercion_curves, propagated_modes):

    # signal_to_fft = pad_timetraces_zeroes(dispersion[0], dispersion[1])
    # signal_to_fft = dispersion
    # signal_after_fft = np.fft.rfft(signal_to_fft[1])
    signal_after_fft = np.fft.rfft(dispersion[1])
    # freq_sampling = time_x_freq[3]
    # time = time_x_freq[0]
    time = dispersion[0]
    # frequency_from_numpy = np.fft.rfftfreq(len(signal_to_fft[1]))*1e4
    frequency_from_numpy = np.fft.rfftfreq(len(dispersion[1]))*1e4
    dt = time[-1]/len(time)
    new_freq_sampling_kHz = frequency_from_numpy
    modes = []
    for ind in range(len(propagated_modes)):
        modes.append(dispercion_curves.getMode(ind))
    # dispercion_curves_of_propagated_mode = KrzyweDyspersji.getMode(0)
    k_vect = dispercion_curves.k_v

#-------------------------------Tutaj kończy się program ---------------------------------

    # new_freq_sampling = np.linspace(freq_sampling[0], freq_sampling[-1], len(signal_after_fft))
    # new_freq_sampling_kHz = new_freq_sampling*1e-3

    G_w = np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2)
    print("Trzeci plot")
    plt.plot(new_freq_sampling_kHz, G_w, '*')
    plt.show()
    print("Czwarty plot")
    plt.plot(frequency_from_numpy, G_w, '*')
    plt.show()
#------------------Wyliczanie ograniczeń --------------------------------
    k_nyq = calculate_k_nyquist(dispercion_curves, dt)
    delta_x = calculate_delta_x(k_nyq)
    delta_k = calculate_delta_k(dispercion_curves, time[-1]) * 0.5e5
    n = calculate_n(k_nyq, delta_k) # n to długość wektora x, liczba próbek na odległości
#--------------------------to już skopiowane i działa-----------------
    if len(modes) > 1:
        mean_data = calculate_mean_mode(dispercion_curves, propagated_modes)
        mean_mode = mean_data[0]
        mean_k_vector = mean_data[1]
    else:
        mean_mode = modes[0]
        mean_k_vector = dispercion_curves.k_v
    mode_0 = mean_mode
    k_vect = mean_k_vector
    plt.plot(mean_mode.all_omega_khz, mean_k_vector)
    plt.show()
    max_k = find_max_k(mode_0, k_vect, new_freq_sampling_kHz[-1])
    new_k_sampling_rad_m = []
    k = 0
    print("Max k = " + str(max_k) + "delta k = " + str(delta_k))
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
        dist_vect.append(i*dx*2*np.pi/len(propagated_modes))

    plt.plot(dist_vect, h_x)
    plt.show()


if __name__ == "__main__":
    KrzyweDyspersji=selectMode.SelectedMode('../eig/kvect', '../eig/omega')
    KrzyweDyspersji.selectMode()
    dist = 2 # w metrach

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()
    # for i in range(length):
    print("Zaraz będzie się dzało :o")
    # make_dispersion_in_bar(length, len(plane), dx, KrzyweDyspersji)
    dispersion = Anim_dyspersji.draw_time_propagation(signal_array, time_x_freq, dist, KrzyweDyspersji)
    print("Pierwszy plot")
    plt.plot(time_x_freq[0], signal_array[3])
    plt.show()
    print("Drugi plot")
    plt.plot(dispersion[0], dispersion[1])
    plt.show()

    gupi_Wilcox_Bedzie_dzialac(dispersion, KrzyweDyspersji, [0, 1, 2, 3])

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
