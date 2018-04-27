import numpy as np
import matplotlib.pyplot as plt
from MES_dir import readData as rd
from MES_dir import dispersion, mode_sampling
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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


def draw_chirp_and_window(signal_array, time_x_freq_array):
    chirp = signal_array[0]
    f_chirp = signal_array[1]
    hanning = signal_array[2]
    chirp_windowed = signal_array[3]
    f_chirp_windowed = signal_array[4]

    time = time_x_freq_array[0]
    freq_sampling = time_x_freq_array[3]

    plt.figure("Chirp z oknem Hanninga - t")

    plt.subplot(321)
    plt.plot(time, chirp)
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")


    plt.subplot(322)
    plt.plot(freq_sampling*1e-3, np.sqrt(f_chirp.real**2 + f_chirp.imag**2))
    plt.title("Chirp DFT")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(323)
    plt.plot(time, hanning)
    plt.title("Okno Hanninga")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(324)
    plt.plot(freq_sampling*1e-3, np.sqrt(np.fft.rfft(hanning).real**2 + np.fft.rfft(hanning).imag**2))
    plt.title("Okno Hanninga")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(325)
    plt.plot(freq_sampling*1e-3, np.sqrt(np.fft.rfft(hanning).real**2 + np.fft.rfft(hanning).imag**2))
    plt.title("Okno Hanninga")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(325)
    plt.plot(np.fft.rfftfreq(len(signal_array[2])), np.sqrt(f_chirp_windowed.real**2 + f_chirp_windowed.imag**2))
    plt.title("Chirp okienkowany")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(326)
    plt.plot(freq_sampling*1e-3, np.sqrt(f_chirp_windowed.real**2 + f_chirp_windowed.imag**2))
    plt.title("Chirp okienkowany DFT")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.35)


def draw_chirp_in_length(signal_array, time_x_freq_array):
    hanning = signal_array[2]
    x = time_x_freq_array[1]
    freq = time_x_freq_array[2]
    freq_sampling = time_x_freq_array[3]

    Vgr = 500 # no dispersion
    k = (freq / Vgr) * 2 * np.pi
    k_sampling = (freq_sampling / Vgr) * 2 * np.pi
    chirpx = np.sin(k*x)
    chirpx_windowed = chirpx*hanning
    f_chirpx = np.fft.rfft(chirpx_windowed)

    plt.figure("Chirp - x")

    plt.subplot(211)
    plt.plot(x, chirpx_windowed)
    plt.title("Chirp na długości")
    plt.xlabel("Długość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(212)
    plt.plot(k_sampling, np.sqrt(f_chirpx.real**2 + f_chirpx.imag**2))
    plt.title("Chirp DFT")
    plt.xlabel("Liczba falowa [rad/m]")
    plt.ylabel("Amplituda [-]")

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=0.5,
                        wspace=0.35)


def draw_transformation(signal_array, time_x_freq_array):
    chirp_windowed = signal_array[3]

    time = time_x_freq_array[0]


    total_samples = 10*len(time)
    total_time = np.linspace(0, 10*time[-1], total_samples)
    fs = len(time)/time[-1]
    sampling_frequnecy3 = np.linspace(0, fs/2, int(total_samples / 2 + 1))
    chirp_in_time = []
    for i in range(total_samples):
        if i < 1000:
            chirp_in_time.append(chirp_windowed[i])
        else:
            chirp_in_time.append(0)

    f_chirp_in_time = np.fft.rfft(chirp_in_time)

    # k_test = np.linspace(0, 1e6*2*np.pi, 10000)
    omega_test = np.ones(int(total_samples / 2 + 1))*1e7

    f_transformed_chirp_in_time_temp = []
    for f, om, t in zip(f_chirp_in_time, omega_test, total_time):
        f_transformed_chirp_in_time_temp.append(f * np.exp(-1j * om * t))

    f_transformed_chirp_in_time = np.array(f_transformed_chirp_in_time_temp)

    transformed_chirp_in_time = np.fft.irfft(f_transformed_chirp_in_time)

    plt.figure("Transformacja")

    plt.subplot(511)
    plt.plot(total_time, chirp_in_time)
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(512)
    plt.plot(sampling_frequnecy3*1e-3, np.sqrt(f_chirp_in_time.real**2 + f_chirp_in_time.imag**2))
    # plt.plot(sampling_frequnecy3, f_chirp_in_time.real)
    plt.title("Chirp DFT")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(513)
    plt.plot(sampling_frequnecy3*1e-3, np.sqrt(f_transformed_chirp_in_time.real**2 + f_transformed_chirp_in_time.imag**2))
    # plt.plot(sampling_frequnecy3, f_transformed_chirp_in_time.real)
    plt.title("Chirp DFT po przekształceniu")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(514)
    plt.plot(total_time, transformed_chirp_in_time)
    plt.title("Chirp po przekształceniu")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(515)
    plt.plot(sampling_frequnecy3*1e-3, f_transformed_chirp_in_time.real
             - f_chirp_in_time.real)
    plt.title("Re widma przeksztłconego - Re widma")
    plt.xlabel("Częstotliwość [kHz]")
    plt.ylabel("Amplituda [-]")

    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)


def get_phase_velocity_sampled(mode, freq_samples):
    omega = rd.read_complex_omega("../eig/omega", mode)
    freq = omega.real / (2 * np.pi)
    kvect = rd.read_kvect("../eig/kvect")
    phase_vel = omega.real/kvect
    return mode_sampling.curve_sampling(freq, phase_vel, freq_samples)


def draw_length_propagation(signal_array, time_x_freq_array):
    chirp_windowed = signal_array[3]

    time = time_x_freq_array[0]

    freq_temp = [f*1000 for f in range(5002)]
    omega = np.array(freq_temp) * 2 * np.pi
    total_samples = 10 * len(time)
    total_time = np.linspace(0, 10 * time[-1], total_samples)
    fs = len(time) / time[-1]
    chirp_in_time = []
    for i in range(total_samples):
        if i < 1000:
            chirp_in_time.append(chirp_windowed[i])
        else:
            chirp_in_time.append(0)

    f_chirp_in_time = np.fft.rfft(chirp_in_time)

    plt.figure("Przebieg w czasie")

    plt.subplot(511)
    plt.plot(total_time, np.fft.irfft([f * np.exp(-1j * om * 1e-4) for f, om in zip(f_chirp_in_time, omega.real)]))
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(512)
    plt.plot(total_time, np.fft.irfft([f * np.exp(-1j * om * 2 * 1e-4) for f, om in zip(f_chirp_in_time, omega.real)]))
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(513)
    plt.plot(total_time, np.fft.irfft([f * np.exp(-1j * om * 3 * 1e-4) for f, om in zip(f_chirp_in_time, omega.real)]))
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(514)
    plt.plot(total_time, np.fft.irfft([f * np.exp(-1j * om * 5 * 1e-4) for f, om in zip(f_chirp_in_time, omega.real)]))
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(515)
    plt.plot(total_time, np.fft.irfft([f * np.exp(-1j * om * 9 * 1e-4) for f, om in zip(f_chirp_in_time, omega.real)]))
    plt.title("Chirp")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda [-]")


    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)


def draw_time_propagation(signal_array, time_x_freq_array):
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

    v_p = get_phase_velocity_sampled(1, np.array(freq_temp))
    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    f_chirp_in_length = f_chirp_in_time
    # f_chirp_in_length = v_p * f_chirp_in_time

    kvect_transform = omega / v_p
    omega_mode = rd.read_complex_omega("../eig/omega", 0)
    kvect = rd.read_kvect("../eig/kvect")
    k1 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 1)
    k2 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 2)
    k3 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 3)
    k4 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)

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

    plt.subplot(512)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(513)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 2) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(514)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 5) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(515)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 7) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")


    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)


def draw_time_propagation_frequency(signal_array, time_x_freq_array):
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

    v_p = get_phase_velocity_sampled(1, np.array(freq_temp))
    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    f_chirp_in_length = f_chirp_in_time
    # f_chirp_in_length = v_p * f_chirp_in_time

    kvect_transform = omega / v_p
    omega_mode = rd.read_complex_omega("../eig/omega", 0)
    kvect = rd.read_kvect("../eig/kvect")
    k1 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 1)
    k2 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 2)
    k3 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 3)
    k4 = 1000* mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)

    plt.figure("Widmo - sing around")

    amp_modulator1 = np.array([2 * np.exp(-(f-1e4)**2 * 1e-7) for f in freq_temp])
    amp_modulator2 = np.array([2 * np.exp(-(f-2 * 1e4)**2 * 1e-7) for f in freq_temp])
    amp_modulator3 = np.array([2 * np.exp(-(f-5 * 1e4)**2 * 1e-7) for f in freq_temp])
    amp_modulator4 = np.array([2 * np.exp(-(f-8 * 1e4)**2 * 1e-7) for f in freq_temp])

    def resonance(iterations, f_chirp_in_time, k, amp_modulator):
        result = f_chirp_in_time

        for i in range(iterations):
            temp_result = []
            for f, kk, amp in zip(result, k, amp_modulator):
                temp_result.append(f * np.exp(-1j * kk * 1) * amp)
            result = temp_result
        return result

    iterations = 10
    f_chirp_in_time1 = resonance(iterations, f_chirp_in_time, k1, amp_modulator1)
    f_chirp_in_time2 = resonance(iterations, f_chirp_in_time, k2, amp_modulator2)
    f_chirp_in_time3 = resonance(iterations, f_chirp_in_time, k3, amp_modulator3)
    f_chirp_in_time4 = resonance(iterations, f_chirp_in_time, k4, amp_modulator4)

    f_temp = []
    for f1, f2, f3, f4 in zip(f_chirp_in_time1, f_chirp_in_time2, f_chirp_in_time3, f_chirp_in_time4):
        f_temp.append(f1 + f2 + f3 + f4)

    f_chirp_in_time_a = np.array(f_temp)

    modul = np.sqrt(f_chirp_in_time_a.real**2 + f_chirp_in_time_a.imag**2)

    plt.subplot(211)
    plt.plot(freq_temp, np.sqrt(f_chirp_in_time.real**2 + f_chirp_in_time.imag**2))
    plt.title("Chirp")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(212)
    plt.plot(freq_temp, modul)
    plt.title("Chirp")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [-]")


    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)

signal_array, time_x_freq = get_chirp()
draw_chirp_and_window(signal_array, time_x_freq)
draw_chirp_in_length(signal_array, time_x_freq)
draw_transformation(signal_array, time_x_freq)
draw_time_propagation(signal_array, time_x_freq)
draw_length_propagation(signal_array, time_x_freq)
draw_time_propagation_frequency(signal_array, time_x_freq)


plt.show()

