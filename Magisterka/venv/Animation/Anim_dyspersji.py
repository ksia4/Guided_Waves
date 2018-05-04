import sympy as sp
import numpy as np
from MES_dir import MES, config, dispersion, mesh, selectMode, mode_sampling
import matplotlib.pyplot as plt
import matplotlib.colors as color

BASECOLOR = (230/255, 230/255, 250/255)


def get_chirp():
    Vgr = 500
    samples = 1000
    time_end = 1e-4
    f_max = 1e5
    length = Vgr*time_end
    x = np.linspace(0, length, samples)
    time = np.linspace(0, time_end, samples)
    freq = np.linspace(0, f_max, samples)

    fs = (samples / time_end)
    freq_sampling = np.linspace(0, fs/2, int(samples/2 + 1))


    chirp = np.sin(2*np.pi*freq*time)
    f_chirp = np.fft.rfft(chirp)

    # freq_sampling = np.fft.fftfreq(chirp.size, d=time_end/samples)

    hanning = []
    for n in range(len(time)):
        hanning.append(0.5*(1-np.cos(np.pi*2*(n + 1)/len(time))))

    chirp_windowed = chirp*hanning
    f_chirp_windowed = np.fft.rfft(chirp_windowed)
    return np.array([chirp, f_chirp, np.array(hanning), chirp_windowed, f_chirp_windowed]),\
           np.array([time, x, freq, freq_sampling])


def find_linear_fc(x1, x2, y1, y2):
    a = (y1 - y2)/(x1 - x2)
    b = y1 - a * x1
    return np.array([b, a]) # od najmniejszej potegi zmiennej

def curve_sampling(omega, values, freq_sampled):
    k = [] # szukane k dla zadanych czestotliwosci -> frequencies

    for om in freq_sampled:

        for i in range(len(omega) - 1):
            if om > omega[i].real and om < omega[i + 1].real:
                linera_fc = find_linear_fc(omega[i].real, omega[i + 1].real, values[i], values[i + 1])
                sampled_k = linera_fc[0] + om * linera_fc[1]
                k.append(sampled_k)
                break
        if om <= omega[0]:
            k.append(values[0])
        if om >= omega[-1]:
            k.append(values[-1])
    return np.array(k)

def curve_sampling_new(omega, values, freq_sampled):
    k = [] # szukane k dla zadanych czestotliwosci -> frequencies

    for om in freq_sampled:
        if om <= omega[0]:
            if omega[0] < 5:
                linear_fc = find_linear_fc(0, omega[0], 0, values[0])
                sampled_k = linear_fc[0] + om * linear_fc[1]
                k.append(sampled_k)
                continue
            else:
                k.append(0)
        if om >= omega[-1]:
            linear_fc = find_linear_fc(omega[-2], omega[-1], values[-2], values[-1])
            sampled_k = linear_fc[0] + om * linear_fc[1]
            k.append(sampled_k)
            continue

        for i in range(len(omega) - 1):
            if om > omega[i].real and om < omega[i + 1].real:
                linera_fc = find_linear_fc(omega[i].real, omega[i + 1].real, values[i], values[i + 1])
                sampled_k = linera_fc[0] + om * linera_fc[1]
                k.append(sampled_k)
                break
            if om == omega[i]:
                k.append(values[i])
    return np.array(k)

def disp_in_time(distance, signal, time, freq, x, KrzyweDyspersji, num_of_modes=4):

    print("Zaczynam liczyć dyspersje")

    freq_temp = [f*100 for f in range(50001)]
    omega = np.array(freq_temp) * 2 * np.pi
    total_samples = 100 * len(time)
    total_time = np.linspace(0, 10 * time[-1], total_samples)
    fs = len(time) / time[-1]
    chirp_in_time = []
    for i in range(total_samples):
        if i < 1000:
            chirp_in_time.append(signal[i])
        else:
            chirp_in_time.append(0)

    total_length = np.linspace(0, 100*x[-1], total_samples)

    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    all_k=[]
    k_vect=KrzyweDyspersji.k_v
    for i in range(num_of_modes):
        temp_mode=KrzyweDyspersji.getMode(i)
        temp_k=[]
        omegi = []
        for om in temp_mode.points:
            omegi.append(om.w)
        all_k.append(curve_sampling(omegi, k_vect, np.array(freq_temp).real))

    print("Zaraz zrobie plota")


    #ZMIENIĆ!!!! ŻEBY MOŻNA BYŁO WIĘCEJ?MNIEJ NIŻ 4 MODY.....
    results = [total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * distance) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])])]
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * distance) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])]))
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")
    plt.plot(results[0], results[1])

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)
    plt.show()
    # print("Koniec funkcji, zwracam!")
    return results

def draw_time_propagation(signal_array, time_x_freq_array, distance, krzyweDyspersji):
    print("wszedłem")
    chirp_windowed = signal_array[3]

    time = time_x_freq_array[0]
    x = time_x_freq_array[1]

    freq_temp = [f*100 for f in range(50001)]
    omega = np.array(freq_temp) * 2 * np.pi
    total_samples = 100 * len(time)
    total_time = np.linspace(0, 10 * time[-1], total_samples)
    fs = len(time) / time[-1]
    chirp_in_time = []
    for i in range(total_samples):
        if i < 1000:
            chirp_in_time.append(chirp_windowed[i])
        else:
            chirp_in_time.append(0)

    total_length = np.linspace(0, 100*x[-1], total_samples)

    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    f_chirp_in_length = f_chirp_in_time

    k_vect = krzyweDyspersji.k_v
    all_k = []
    for m in range(4): # POPRAWIĆ
        mode = krzyweDyspersji.getMode(m)
        omegi = mode.allOmega
        all_k.append(mode_sampling.curve_sampling(np.array(omegi).real, np.array(k_vect).real, np.array(freq_temp).real))

    # plt.figure("Przebieg w odleglosci")
    #
    # plt.subplot(511)
    results = [total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * distance) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])])]
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * distance) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])]))

    # plt.plot(results[0], results[1])
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")

    # plt.subplot(512)
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])]))
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")
    #
    # plt.subplot(513)
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 2) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, all_k[0], all_k[1], all_k[2], all_k[3])]))
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")
    #
    # plt.subplot(514)
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 5) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, all_k[0], all_k[1], all_k[2], all_k[3])]))
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")
    #
    # plt.subplot(515)
    # plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 7) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, all_k[0], all_k[1], all_k[2], all_k[3])]))
    # plt.title("Chirp")
    # plt.xlabel("Odległość [m]")
    # plt.ylabel("Amplituda [-]")


    # plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
    #                     wspace=0.35)
    #
    # plt.show()
    return results



def make_disp(chirp, time_vect, x_vect, freq_vect, numb_of_planes, distance_between_planes, KrzyweDyspersji):
    time_trace = []
    time = []
    for i in range(numb_of_planes):
        temp=disp_in_time(i*distance_between_planes, chirp, time_vect, freq_vect, x_vect, KrzyweDyspersji)
        single_plane_time_trace = temp[1]
        if i == 0:
            time = temp[0]
        time_trace.append(single_plane_time_trace)
    return time_trace


def make_chirp(minf, maxf, duration_time, Hanning):
    Vgr = 500
    samples = 1000
    time_end = duration_time
    f_max = maxf
    f_min = minf
    length = Vgr*time_end
    x = np.linspace(0, length, samples)
    time = np.linspace(0, time_end, samples)
    freq = np.linspace(f_min, f_max, samples)

    fs = (samples / time_end)
    freq_sampling = np.linspace(0, fs/2, int(samples/2 + 1))


    chirp = np.sin(2*np.pi*freq*time)
    f_chirp = np.fft.rfft(chirp)

    # freq_sampling = np.fft.fftfreq(chirp.size, d=time_end/samples)

    hanning = []
    for n in range(len(time)):
        hanning.append(0.5*(1-np.cos(np.pi*2*(n + 1)/len(time))))

    chirp_windowed = chirp*hanning
    f_chirp_windowed = np.fft.rfft(chirp_windowed)
    if Hanning:
        return np.array(chirp_windowed),\
        np.array([time, x, freq])
    else:
        return np.array(chirp),\
        np.array([time, x, freq])

def draw_bar(vertices, num_of_points_in_one_piece, length):
    print(len(vertices))
    planes = len(vertices)/num_of_points_in_one_piece
    check = int(planes)
    if planes-check != 0:
        print("Coś tu się źle policzyło...")
        exit(0)
    print(check)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=BASECOLOR)

    # ax.scatter(vertices[0:num_of_points_in_one_piece, 0], vertices[0:num_of_points_in_one_piece, 1], vertices[0:num_of_points_in_one_piece, 2], color='red')
    # ax.scatter(vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 0], vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 1], vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 2], color='white', edgecolor='red')

    lim = int(length/2)
    ax.set_xlim([-1, length+1])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    plt.show()

def animDisp(vertices, one_piece, length):
    print(len(vertices))
    print("Teraz się będą animacje robić :o")
    planes = len(vertices) / one_piece
    check = int(planes)
    if planes-check != 0:
        print("Coś tu się źle policzyło...")
        exit(0)
    print(check)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    lim = int(length/2)
    ax.set_xlim([-1, length+1])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    #Po chwilach czasowych:
    # for t in range(len(traces[0])):
    #     signal = traces[:][t]
    #     #Po sygnale - mapujemy kolor
    #     # color=[]
    #     color = BASECOLOR
    #     for ind, s in enumerate(signal):
    #         if ind == len(signal) - 1:
    #             break
    #         if s >= 0:
    #             color = (1, 1-s, 1-s)
    #             # color.append((1, 1-s, 1-s))
    #         else:
    #             color = (1+s, 1+s, 1)
    #             # color.append((1+s, 1+s, 1))
    #         ax.scatter(vertices[ind * one_piece:(ind + 1) * one_piece, 0], vertices[ind * one_piece:(ind + 1) * one_piece, 1], vertices[ind * one_piece:(ind + 1) * one_piece, 2], c=color)
    #
    #     name=str(t)+'.png'
    #     plt.savefig(name)
    #     # plt.show()

    # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=BASECOLOR)

    ax.scatter(vertices[0:one_piece, 0], vertices[0:one_piece, 1], vertices[0:one_piece, 2], color='red')
    ax.scatter(vertices[one_piece:2*one_piece, 0], vertices[one_piece:2*one_piece, 1], vertices[one_piece:2*one_piece, 2], color='white', edgecolor='red')


    plt.show()

def make_dispersion_in_bar(num_of_planes, one_piece, dx, KrzyweDyspersji):
    signal_array, time_x_freq = get_chirp()
    all_signals = []
    time = []
    for i in range(num_of_planes):
        print(i)
        all_data = draw_time_propagation(signal_array, time_x_freq, i*dx, KrzyweDyspersji)
        if i == 0:
            time = all_data[0]
        all_signals.append(all_data[1])
        # plt.plot(all_signals[i][0], all_signals[i][1])
        # plt.show()
    # print("Tyle mamy sygnałów, powinno być 100:")
    # print(len(all_signals))
    # print("Tyle mamy próbek w jednym sygnale")
    # print(len(all_signals[0]))
    # print("taki mamy wektor czasu")
    # print(time)
    # exit(0)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # lim = int(length/2)
    # ax.set_xlim([-1, length+1])
    # ax.set_ylim([-lim, lim])
    # ax.set_zlim([-lim, lim])
    for t in range(int(len(time)*0.4)): # dla każdej chwili czasowej 0 - 0.1 s
        print(t)
        if t % 20 != 0:
            continue
        plane_color = (1, 1, 1)
        # actual_signal_0 = all_signals[:][0]
        # print(len(actual_signal_0))
        # exit(0)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        lim = int(length/2)
        ax.set_xlim([-1, length+1])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])
        for plane in range(len(all_signals)): # dla każdej płaszczyzny
            actual_signal = all_signals[plane][t]
            if actual_signal >= 0:
                plane_color = (1, 1-actual_signal, 1-actual_signal)
            else:
                plane_color = (1+actual_signal, 1+actual_signal, 1)

            # if abs(actual_signal)>0.5:
            # print("plotuje bo wartość sygnału to")
            # print(actual_signal)

            ax.scatter(vertices[plane*one_piece: (plane+1)*one_piece-1, 0], vertices[plane*one_piece: (plane+1)*one_piece-1, 1], vertices[plane*one_piece: (plane+1)*one_piece-1, 2], c=plane_color)
        # plt.show()
        # exit(0)
        name = str(t) + '.png'
        plt.savefig(name)
        print("Zapisano kolejny obraz")
        print(name)
        # plt.show()

            # else:
            #     print("Wartość sygnału")
            #     print(actual_signal)

        # #Po sygnale - mapujemy kolor
        # # color=[]
        # color = BASECOLOR
        # for ind, s in enumerate(signal):
        #     if ind == len(signal) - 1:
        #         break
        #     if s >= 0:
        #         color = (1, 1-s, 1-s)
        #         # color.append((1, 1-s, 1-s))
        #     else:
        #         color = (1+s, 1+s, 1)
        #         # color.append((1+s, 1+s, 1))
        #     ax.scatter(vertices[ind * one_piece:(ind + 1) * one_piece, 0], vertices[ind * one_piece:(ind + 1) * one_piece, 1], vertices[ind * one_piece:(ind + 1) * one_piece, 2], c=color)


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
    plane = mesh.circle_mesh_full(1, radius, num_of_circles, num_of_points_at_c1)
    vertices = mesh.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    draw_bar(vertices, len(plane), length)
    KrzyweDyspersji=selectMode.SelectedMode('../eig/kvect', '../eig/omega')
    KrzyweDyspersji.selectMode()
    # KrzyweDyspersji.plot_modes(50)

    # signal_array, time_x_freq = get_chirp()
    # for i in range(length):
    print("Zaraz będzie się dzało :o")
    make_dispersion_in_bar(length, len(plane), dx, KrzyweDyspersji)
        # dispersion = draw_time_propagation(signal_array, time_x_freq, i*dx, KrzyweDyspersji)

    # chirp, time_x_frq = make_chirp(0, 1e5, 1e-4, True)
    # timeTraces = make_disp(chirp, time_x_frq[0], time_x_frq[1], time_x_frq[2], length, dx, KrzyweDyspersji)
    # animDisp(vertices, len(plane), length)

    #
    # signal_array, time_x_freq = chirp_propagation.get_chirp()

