from Propagation import selectMode
from Animation import Anim_dyspersji
import matplotlib.pyplot as plt
import numpy as np

def find_accurate_len(actual_len, factor=8):
    min_len = factor * actual_len
    estimated_len = 256
    while estimated_len < min_len:
        estimated_len *= 2
    return estimated_len

def pad_timetraces_zeroes(time_vector, signal_vector, multi=8):
    actual_len = len(signal_vector)
    print("teraz było tyle punktów")
    estimated_len = find_accurate_len(actual_len, multi)
    print(actual_len)
    print("teraz będzie o tyle punktów")
    dt = time_vector[-1]/len(time_vector)
    print("O takie jest dt " + str(dt))
    print("A o tyle dołożymy na koniec")
    print(estimated_len - actual_len)
    len_to_add = estimated_len - actual_len
    new_signal = []
    new_time_vector = []
    for i in range(len(signal_vector)):
        new_signal.append(signal_vector[i])
        new_time_vector.append((time_vector[i]))
    for i in range(len_to_add):
        new_signal.append(0)
        new_time_vector.append(new_time_vector[-1] + dt)

    plt.plot(new_time_vector, new_signal)
    plt.show()
    return [new_time_vector, new_signal]

def calculate_n(k_Nyquista, delta_k, factor=1.1):

    # Funkcja obliczająca porządaną ilość próbek w wektorze odległości w końcowym śladzie
    # powinien być zbliżony do długości wektora czasu
    # n > 2(k_Nyquista/deltak)
    if factor <= 1:
        print("Współczynnik musi być większy od 1, przyjęta wartość wynosi 1,1")
        factor = 1.1

    return int(factor * 2 * (k_Nyquista/delta_k))

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

def calculate_delta_k(max_v_gr, signal_duration, factor=0.9):
    # delta k powinno być = 1/(n(delta_x) i mniejsze niż 1/(m*delta_t*v_gr_max) m*delta_t jest równe długości trwania sygnału :)
    if signal_duration <= 0:
        print("Długość sygnału musi być większa od 0")
        exit(0)
    if factor >= 1:
        print("Współczynnik musi być mniejszy od 1, przyjęta wartość to 0,9")
        factor = 0.9
    # for mode in dispercion_curves.AllModes.modeTable:
    #     temp_v_gr = 1000 * max(disp.calculate_group_velocity(mode.all_omega_khz, dispercion_curves.k_v/1000))
    #     if temp_v_gr > max_v_gr:
    #         max_v_gr = temp_v_gr
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
            calc_k = Anim_dyspersji.curve_sampling_new(modes[mode_ind + 1].all_omega_khz, dispercion_curves.k_v, [omegs[ind]])[0]
            mean_k = mean_k + calc_k
        mean_k = mean_k/(len(modes))
        mean_mode.addPoint(modes[0].points[ind])
        mean_mode.points[ind].k = mean_k
        mean_k_vector.append(mean_k)
    plt.plot(mean_mode.all_omega_khz, mean_k_vector)
    plt.show()
    return [mean_mode, mean_k_vector]

def Wilcox_compensation2(dispersion, dispercion_curves, propagated_modes, need_to_pad = False):
    if need_to_pad:
        signal_to_fft = pad_timetraces_zeroes(dispersion[0], dispersion[1])
    else:
        signal_to_fft = dispersion
    # signal_to_fft = dispersion
    # signal_after_fft = np.fft.rfft(signal_to_fft[1])
    # signal_after_fft = np.fft.rfft(dispersion[1])
    signal_after_fft = np.fft.rfft(signal_to_fft[1])
    # freq_sampling = time_x_freq[3]
    # time = time_x_freq[0]
    # time = dispersion[0]
    time = signal_to_fft[0]
    dt = time[-1]/len(time)
    frequency_from_numpy = np.fft.rfftfreq(len(signal_to_fft[1]), d=dt)*1e-3
    # frequency_from_numpy = np.fft.rfftfreq(len(dispersion[1]))*1e4

    new_freq_sampling_kHz = frequency_from_numpy
    modes = []
    for ind in range(len(propagated_modes)):
        modes.append(dispercion_curves.getMode(ind)) #było samo ind
    # dispercion_curves_of_propagated_mode = KrzyweDyspersji.getMode(0)
    k_vect = dispercion_curves.k_v

    # new_freq_sampling = np.linspace(freq_sampling[0], freq_sampling[-1], len(signal_after_fft))
    # new_freq_sampling_kHz = new_freq_sampling*1e-3

    G_w = np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2)
    print("Trzeci plot")
    plt.plot(new_freq_sampling_kHz, G_w, '*')
    plt.show()
    print("Czwarty plot")
    plt.plot(frequency_from_numpy, G_w, '*')
    plt.show()

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
    v_gr_max = 0
    # test = []
    for ind in range(len(k_vect) - 1):
        print("Indeks wynosi " + str(ind))
        value = (mode_0.points[ind + 1].wkat_complex - mode_0.points[ind].wkat_complex)/(k_vect[ind+1]-k_vect[ind])
        # test.append(value)
        if value > v_gr_max:
            v_gr_max = value
    # plt.plot(test)
    # plt.show()

    #------------------Wyliczanie ograniczeń --------------------------------
    k_nyq = calculate_k_nyquist(dispercion_curves, dt)
    delta_x = calculate_delta_x(k_nyq)
    delta_k = calculate_delta_k(v_gr_max.real, time[-1])

    n = calculate_n(k_nyq, delta_k) # n to długość wektora x, liczba próbek na odległości

    max_k = find_max_k(mode_0, k_vect, new_freq_sampling_kHz[-1])
    new_k_sampling_rad_m = []
    while max_k/delta_k > 40000:
        delta_k = delta_k * 5
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

    h_x = np.fft.ifft(H_k) / (2000 * np.pi)
    distance = 1/delta_k # w metrach
    n = len(h_x)
    dx = distance/n
    print("dx wynosi " + str(dx) + "A cały dystans " + str(distance))
    dist_vect = []
    for i in range(n):
        dist_vect.append(i*dx*2*np.pi/len(propagated_modes))

    plt.plot(dist_vect, h_x)
    plt.show()

def wave_length_propagation(signal, numbers_of_modes, disp_curves, distance_m, F_PADZEROS, mult=8):
    modes_table = []
    for mode_number in numbers_of_modes:
        modes_table.append(disp_curves.getMode(mode_number))

    if F_PADZEROS:
        signal_to_fft = pad_timetraces_zeroes(signal[0], signal[1], mult)
    else:
        signal_to_fft = signal

    signal_after_fft = np.fft.rfft(signal_to_fft[1])
    time = signal_to_fft[0]
    dt = time[-1]/len(time)
    frequency_from_numpy = np.fft.rfftfreq(len(signal_to_fft[1]), d=dt)*1e-3#*1e4

    print("Plotuje po fftxD")
    plt.plot(frequency_from_numpy, np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2))
    plt.show()

    k_vect = []
    new_signal_after_fft = []
    print("Długość pętli")
    print(len(frequency_from_numpy))
    for ind, f in enumerate(frequency_from_numpy):
        if ind%100 == 0:
            print(ind)
        k_vect.append(0)
        for mode in modes_table:
            k_vect[-1] += mode.findKWithGivenOmega_kHz(f)
        new_signal_after_fft.append(signal_after_fft[ind] * np.exp(-1j * k_vect[ind] * distance_m))

    plt.plot(frequency_from_numpy, np.sqrt(np.array(new_signal_after_fft).real**2 + np.array(new_signal_after_fft).imag**2))
    plt.show()

    propagated_signal = np.fft.irfft(new_signal_after_fft) #/distance_m
    print("Przepropagowałam sygnał?")
    print(len(propagated_signal))
    print(len(time))
    new_time = np.linspace(time[0], time[-1], len(propagated_signal))
    plt.plot(new_time, propagated_signal)
    plt.show()
    return [new_time, propagated_signal]

#Te funkcje chce poprawić tak, żeby przyjmowała sygnał który mamy zrobić do kompensacji i odległość po jakiej się ma kompensować i ile modów ma propagować :)
def time_reverse_compensation(signal, zeros=0.01):
    time_vector = signal[0]
    new_signal = []
    for s in signal[1]:
        new_signal.append(s)
    new_signal.reverse()
    print("Odwrócony sygnał ma być o tu")
    plt.plot(time_vector, new_signal)
    plt.show()
    # temp_signal = []
    # temp_signal2 = []
    # for s in signal[1]:
    #     temp_signal2.append(s)
    #     if abs(s) > zeros:
    #         temp_signal.append(s)
    #     else:
    #         temp_signal.append(0)
    #
    # temp_signal.reverse()
    # temp_signal2.reverse()
    # new_signal = []
    # for ind, s in enumerate(temp_signal):
    #     if s != 0:
    #         new_signal.append(temp_signal2[ind])
    #
    # time_vector = signal[0][0:len(new_signal)]
    return [time_vector, new_signal]

def linear_mapping_compensation(signal, numbers_of_modes, disp_curves):

    signal_after_fft = np.fft.rfft(signal[1])
    time = signal[0]
    dt = time[-1]/len(time)
    frequency_from_numpy = np.fft.rfftfreq(len(signal[1]), d=dt)*1e-3
    G_w = np.sqrt(signal_after_fft.real**2 + signal_after_fft.imag**2)
    #znalezienie najsilniejszej/średniej omegi

    max_g = G_w[0]
    max_ind = 0
    for ind, g in enumerate(G_w):
        if g>max_g:
            max_g = g
            max_ind = ind

    w_0 = frequency_from_numpy[max_ind]

    plt.plot(frequency_from_numpy, G_w)
    plt.show()

    print("Max silna omega w k_Hz" + str(frequency_from_numpy[max_ind]))

    modes = []
    for ind in range(len(numbers_of_modes)):
        modes.append(disp_curves.getMode(numbers_of_modes[ind]))
    if len(modes) > 1:
        mean_data = calculate_mean_mode(disp_curves, numbers_of_modes)
        mean_mode = mean_data[0]
        mean_k_vector = mean_data[1]
    else:
        mean_mode = modes[0]
        mean_k_vector = disp_curves.k_v

    k_vect = []
    for ind, f in enumerate(frequency_from_numpy):
        k_vect.append(mean_mode.findKWithGivenOmega_kHz(f))
    G_k = []
    ind = 0
    print("Teraz będziemy robić G(k)")
    print(len(k_vect))
    for temp_k in k_vect:
        om = find_omega_in_dispercion_curves(mean_mode, temp_k, mean_k_vector)
        val = find_value_by_omega_in_G_w(signal_after_fft, frequency_from_numpy, om)
        G_k.append(val)

    k_0 = mean_mode.findKWithGivenOmega_kHz(w_0)
    k_1 = 0
    point1 = selectMode.Point()
    point2 = selectMode.Point()
    point3 = selectMode.Point()
    if w_0 < mean_mode.minOmega:
        k_1 = 0
    if w_0 > mean_mode.points[-1].w:
        point1 = mean_mode.points[-2]
        point2 = mean_mode.points[-1]
    for ind in range(len(mean_mode.points) - 1):
        if mean_mode.points[ind].w == w_0:
            point1 = mean_mode.points[ind-1]
            point2 = mean_mode.points[ind]
            point3 = mean_mode.points[ind+1]
            break

        if mean_mode.points[ind].w > w_0:
            continue
        if mean_mode.points[ind].w < w_0 and mean_mode.points[ind+1].w > w_0:
            point1 = mean_mode.points[ind]
            point2 = mean_mode.points[ind+1]
            break
    if point3.k == 0:
        k_1 = (point1.k - point2.k)/(point1.w - point2.w)
    else:
        k_1_left = (point1.k - point2.k)/(point1.w - point2.w)
        k_1_right = (point2.k - point3.k)/(point2.w - point3.w)
        k1 = (k_1_right + k_1_left)/2

    new_G_w = []
    print("Liczę nowe G(w) to z falką")

    for ind, f in enumerate(frequency_from_numpy):
        print(len(frequency_from_numpy)-ind)
        k_lin = k_0 + k_1*(f - w_0)
        val = find_value_by_omega_in_G_w(G_k, frequency_from_numpy, k_lin)
        new_G_w.append(val)

    new_g_t = np.fft.ifft(new_G_w)
    new_time = np.linspace(time[0], time[-1], len(new_g_t))

    print("Usunęłam dyspersję")
    plt.plot(new_time, new_g_t)
    plt.show()


        # print(len(k_vect) - ind)
        # ind +=1
    # plt.plot(k_vect, np.sqrt(np.array(G_k).real**2 + np.array(G_k).imag**2))
    # plt.show()




if __name__ == "__main__":
    KrzyweDyspersji= selectMode.SelectedMode('../eig/kvect', '../eig/omega')
    KrzyweDyspersji.selectMode()
    dist = 2 # w metrach

    signal_array, time_x_freq = Anim_dyspersji.get_chirp()
    # # for i in range(length):
    # print("Zaraz będzie się dzało :o")
    # # make_dispersion_in_bar(length, len(plane), dx, KrzyweDyspersji)
    # dispersion = Anim_dyspersji.draw_time_propagation(signal_array, time_x_freq, dist, KrzyweDyspersji)
    # print("Pierwszy plot")
    # plt.plot(time_x_freq[0], signal_array[3])
    # plt.show()
    # print("Drugi plot")
    # plt.plot(dispersion[0], dispersion[1])
    # plt.show()
    print("wpuszczany sygnał")
    plt.plot(time_x_freq[0], signal_array[3])
    plt.xlabel("time[s]")
    plt.show()
    signal = wave_length_propagation([time_x_freq[0], signal_array[3]], [0, 1, 2, 3], KrzyweDyspersji, dist, True, 100)

    print("kompensacja Wilcoxem")
    Wilcox_compensation2(signal, KrzyweDyspersji, [0, 1, 2, 3])


    # print("Taki jest zdyspersowany sygnał")
    # plt.plot(signal[0], signal[1])
    # plt.xlabel("time[s]")
    # plt.show()

    # linear_mapping_compensation(signal, [0, 1, 2, 3], KrzyweDyspersji)
    # signal = wave_length_propagation([time_x_freq[0], signal_array[3]], [0, 1, 2, 3], KrzyweDyspersji, dist, True, 100)

    # inversed = time_reverse_compensation(signal)
    #
    # print("Kompensujemy o 1 m (1/4 odległości)")
    #
    # compensated1 = wave_length_propagation(inversed, [0, 1, 2, 3], KrzyweDyspersji, 1, True, 4)
    #
    # print("Kompensujemy o 2 m (1/2 odległości)")
    #
    # compensated2 = wave_length_propagation(inversed, [0, 1, 2, 3], KrzyweDyspersji, 2, True, 4)
    #
    # print("Komensujemy o 3 m (3/4 odległości)")
    #
    # compensated3 = wave_length_propagation(inversed, [0, 1, 2, 3], KrzyweDyspersji, 3, True, 4)
    #
    # print("Kompensujemy całościowo o 4 m")
    #
    # compensated4 = wave_length_propagation(inversed, [0, 1, 2, 3], KrzyweDyspersji, 4, True, 4)
    #
    # print("Kompensujemy za bardzo (o 5 m)")
    #
    # compensated5 = wave_length_propagation(inversed, [0, 1, 2, 3], KrzyweDyspersji, 5, True, 4)
    #
    # print("Zestawienie wyników")
    # plt.figure("Porównanie kompensacji")
    #
    # plt.subplot(511)
    # plt.plot(compensated1[0], compensated1[1])
    # plt.title("1 metr")
    # plt.xlabel("czas[s]")
    # plt.ylabel("Amplituda[-]")
    #
    # plt.subplot(512)
    # plt.plot(compensated2[0], compensated2[1])
    # plt.title("2 metr")
    # plt.xlabel("czas[s]")
    # plt.ylabel("Amplituda[-]")
    #
    # plt.subplot(513)
    # plt.plot(compensated3[0], compensated3[1])
    # plt.title("3 metr")
    # plt.xlabel("czas[s]")
    # plt.ylabel("Amplituda[-]")
    #
    # plt.subplot(514)
    # plt.plot(compensated4[0], compensated4[1])
    # plt.title("4 metr")
    # plt.xlabel("czas[s]")
    # plt.ylabel("Amplituda[-]")
    #
    # plt.subplot(515)
    # plt.plot(compensated5[0], compensated5[1])
    # plt.title("5 metr")
    # plt.xlabel("czas[s]")
    # plt.ylabel("Amplituda[-]")
    #
    # plt.show()

