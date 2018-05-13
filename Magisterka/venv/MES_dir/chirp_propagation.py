import numpy as np
import matplotlib.pyplot as plt
from MES_dir import readData as rd
from MES_dir import dispersion, mode_sampling, config, selectMode
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def getChirp(numOfSamples, signalTime, maxFreq, putWindow=True, drawSignal=False):
    def getVelocity():
        # Lame constants

        # mi is equal to Kirchoff moduli
        mi = config.young_mod / (2 * (1 + config.poisson_coef))

        # lambda
        lam = config.young_mod * config.poisson_coef / ((1 - 2 * config.poisson_coef) * (1 + config.poisson_coef))

        # velocity
        velLongitudinal = np.sqrt((lam + 2 * mi) / config.density) / 1000  # [m/s]
        # velShear = np.sqrt(mi / config.density) / 1000  # [m/s]
        return velLongitudinal

    def plotSignal(signalArray):
        time = signalArray[0]
        chirp = signalArray[1]
        freq = signalArray[2]
        f_chirp = signalArray[3]

        plt.figure("Chirp z oknem Hanninga - t")
        # chirp
        plt.subplot(211)
        plt.plot(time, chirp)
        plt.title("Chirp")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda [-]")

        # widmo amplitudowe
        plt.subplot(212)
        plt.plot(freq * 1e-3, np.sqrt(f_chirp.real ** 2 + f_chirp.imag ** 2))
        plt.title("Chirp DFT")
        plt.xlabel("Częstotliwość [kHz]")
        plt.ylabel("Amplituda [-]")


        plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=0.5,
                            wspace=0.35)
        plt.show()

    # vLongitudinal = getVelocity() #depends on material properties
    samples = numOfSamples # 1000
    timeEnd = signalTime # 1e-4
    fMax = maxFreq / 2 # 1e5
    time = np.linspace(0, timeEnd, samples)
    freq = np.linspace(0, fMax, samples)

    fs = (samples / timeEnd) # sampling frequency
    freqSampling = np.linspace(0, fs/2, int(samples/2 + 1)) # samples in freq domain


    chirp = np.sin(2*np.pi*freq*time)
    fChirp = np.fft.rfft(chirp)

    if putWindow:
        hanning = []
        for n in range(len(time)):
            hanning.append(0.5 * (1 - np.cos(np.pi * 2 * (n + 1) / len(time))))

        chirpWindowed = chirp * hanning
        fChirpWindowed = np.fft.rfft(chirpWindowed)
        signalArray = np.array([time, chirpWindowed, freqSampling, fChirpWindowed])
        if drawSignal:
            plotSignal(signalArray)
        return signalArray

    else:
        signalArray = np.array([time, chirp, freqSampling, fChirp])
        if drawSignal:
            plotSignal(signalArray)
        return signalArray


def chirpLengthPropagation(signalArray, numOfModes, length, drawSignal=False):
    def getSignalToPropagate():
        time = signalArray[0]
        chirp = signalArray[1]

        # adding zeroes
        totalSamples = 10 * len(time)
        totalTime = np.linspace(0, 10 * time[-1], totalSamples)  # need it to plot signal
        chirpInTime = []
        for i in range(totalSamples):
            if i < len(time):
                chirpInTime.append(chirp[i])
            else:
                chirpInTime.append(0)

        fChirpInTime = np.fft.rfft(chirpInTime)  # signal to propagate in length

        samplingFreq = totalSamples / (10 * time[-1])

        omega = np.array([i * (samplingFreq / totalSamples) * 2 * np.pi for i in range(int(totalSamples / 2 + 1))])

        return totalTime, omega, fChirpInTime

    def getKForModes(omegaInSpectrum):
        # getting modes
        allModes = selectMode.SelectedMode('../eig/kvect', '../eig/omega')
        allModes.selectMode()
        wantedModes = []
        for i in range(numOfModes):
            wantedModes.append(allModes.getMode(i))
        # modes sampling to get k for omega in signal spectrum
        kForDesiredModes = []
        for mode in wantedModes:
            om = []
            k = []
            for point in mode.points:
                om.append(point.wkat_real_part)
                k.append(point.k)
            kMode = mode_sampling.curve_sampling(om, k, omegaInSpectrum)
            kForDesiredModes.append(kMode)
        return np.array(kForDesiredModes)

    # returns omega~freg of samples and signa in frequency domain
    time, omega, fSignal = getSignalToPropagate()

    # return array of k vectors for all modes in numOfModes
    kForDesiredModes = getKForModes(omega / (2* np.pi))

    resultSignalSpectrum = []
    for ind, fSample in enumerate(fSignal):
        # fTemp = fSample / numOfModes
        # newSignalTemp = 0
        # for kMode in kForDesiredModes:
        #     newSignalTemp += fTemp * np.exp(-1j * kMode[ind] * length)
        newSignalTemp = fSample
        for kMode in kForDesiredModes:
            print(kMode[ind])
            newSignalTemp *= np.exp(-1j * kMode[ind] * length)
        resultSignalSpectrum.append(newSignalTemp)
    resultSignalSpectrum = np.array(resultSignalSpectrum)

    print(kForDesiredModes[0, -10: -1])
    print(np.shape(kForDesiredModes))

    if drawSignal:
        plt.figure("Chirp po propagacji")
        # chirp
        plt.subplot(211)
        plt.plot(time, np.fft.irfft(fSignal))
        plt.title("Chirp")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda [-]")

        # chirp adter prop
        plt.subplot(211)
        plt.plot(time, np.fft.irfft(resultSignalSpectrum))
        plt.title("Chirp")
        plt.xlabel("Czas [s]")
        plt.ylabel("Amplituda [-]")

        # spectrum after prop
        plt.subplot(212)
        plt.plot(omega / (2000 * np.pi), np.sqrt(resultSignalSpectrum.real**2 + resultSignalSpectrum.imag**2))
        plt.title("Chirp spectrum after propagation")
        plt.xlabel("Częstotliwość [kHz]")
        plt.ylabel("Amplituda [-]")

        plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=0.5,
                            wspace=0.35)
        plt.show()

    return time, np.fft.irfft(resultSignalSpectrum) #result signal in time domain

def draw_time_propagation(signalArray):
    def get_phase_velocity_sampled(mode, freq_samples):
        omega = rd.read_complex_omega("../eig/omega", mode)
        freq = omega.real / (2 * np.pi)
        kvect = rd.read_kvect("../eig/kvect")
        phase_vel = omega.real / kvect
        return mode_sampling.curve_sampling(freq, phase_vel, freq_samples)

    chirp_windowed = signalArray[1]

    time = signalArray[0]
    vel = 500
    x = time * vel

    freq_temp = [f*10 for f in range(50001)]
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

    total_length = np.linspace(0, 10*x[-1], total_samples)

    v_p = get_phase_velocity_sampled(1, np.array(freq_temp))
    f_chirp_in_time = np.fft.rfft(chirp_in_time)
    f_chirp_in_length = f_chirp_in_time
    # f_chirp_in_length = v_p * f_chirp_in_time

    kvect_transform = omega / v_p
    omega_mode = rd.read_complex_omega("../eig/omega", 0)
    kvect = rd.read_kvect("../eig/kvect")
    k1 = mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 1)
    k2 = mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 2)
    k3 = mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)
    omega_mode = rd.read_complex_omega("../eig/omega", 3)
    k4 = mode_sampling.curve_sampling(omega_mode.real, kvect.real, np.array(freq_temp).real)

    print(k1[-10: -1])
    print(np.array(freq_temp).real[-10: -1])
    print(omega_mode.real[-10: -1])
    print(kvect.real[-10: -1])
    print(k1[0: 20])
    print(np.shape(k1))
    plt.figure("Przebieg w odleglosci")

    plt.subplot(511)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1000 * 0) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(512)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1000 * 1) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(513)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1000 * 2) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_time, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(514)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1000 * 5) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")

    plt.subplot(515)
    plt.plot(total_length, np.fft.irfft([f * np.exp(-1j * (kk1 + kk2 + kk3 + kk4) * 1000 * 7) for f, kk1, kk2, kk3, kk4 in zip(f_chirp_in_length, k1, k2, k3, k4)]))
    plt.title("Chirp")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Amplituda [-]")


    plt.subplots_adjust(top=0.96, bottom=0.1, left=0.10, right=0.95, hspace=1.4,
                        wspace=0.35)
    plt.show()
if __name__ == "__main__":
    # get_chirp(numOfSamples, signalTime, maxFreq, putWindow=True, drawSignal=False)
    # fc returns list [time, chirp, freqSampling, fChirp]
    chirp = getChirp(1000, 1e-4, 1e5)
    # chirpLengthPropagation(signalArray, numOfModes, drawSignal=False)
    timeVector, ampVector = chirpLengthPropagation(chirp, 4, 2, drawSignal=True)
    # draw_time_propagation(chirp)



