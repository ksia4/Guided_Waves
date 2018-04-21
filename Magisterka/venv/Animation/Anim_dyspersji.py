import sympy as sp
import numpy as np
from MES_dir import MES, config, dispersion, mesh, selectMode
import matplotlib.pyplot as plt
import matplotlib.colors as color

BASECOLOR = (230/255, 230/255, 250/255)


def disp_in_time(distance, signal, time, freq, x, KrzyweDyspersji):
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

    # v_p = get_phase_velocity_sampled(1, np.array(freq_temp))
    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    f_chirp_in_length = f_chirp_in_time
    # f_chirp_in_length = v_p * f_chirp_in_time

    # kvect_transform = omega / v_p
    # omega_mode = rd.read_complex_omega("../eig/omega", 0)
    # kvect = rd.read_kvect("../eig/kvect")
    # k1 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    # omega_mode = rd.read_complex_omega("../eig/omega", 1)
    # k2 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    # omega_mode = rd.read_complex_omega("../eig/omega", 2)
    # k3 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    # omega_mode = rd.read_complex_omega("../eig/omega", 3)
    # k4 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)

    KrzyweDyspersji.
    print(k1[-10: -1])
    print(np.array(freq_temp).real[-10: -1])
    print(omega_mode.real[-10: -1])
    print(kvect.real[-10: -1])
    plt.figure("Przebieg w odleglosci")

    plt.subplot(511)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 0) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")


    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)



def make_disp(chirp, time_vect, x_vect, freq_vect, numb_of_planes, distance_between_planes, KrzyweDyspersji):
    time_trace=[]
    for i in range(numb_of_planes):
        single_plane_time_trace=disp_in_time(i*distance_between_planes, chirp, time_vect, freq_vect, x_vect, KrzyweDyspersji)
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

    # rysowanie wykresow
    # config.show_plane = False
    # config.show_bar = True
    # config.show_elements = False

    # obliczenia
    plane = mesh.circle_mesh_full(1, radius, num_of_circles, num_of_points_at_c1)
    vertices = mesh.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    draw_bar(vertices, len(plane), length)
    KrzyweDyspersji=selectMode.SelectedMode('../eig/kvect', '../eig/omega')
    KrzyweDyspersji.selectMode()
    # KrzyweDyspersji.plot_modes(50)

    chirp, time_x_frq = make_chirp(0, 1e5, 1e-4, True)
    timeTraces=make_disp(chirp, time_x_frq[0], time_x_frq[1], time_x_frq[2], length, dx, KrzyweDyspersji)
    #
    # signal_array, time_x_freq = chirp_propagation.get_chirp()

