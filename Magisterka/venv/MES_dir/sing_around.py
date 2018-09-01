import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from MES_dir import MES, config
from MES_dir import readData as rd

def valueBetweenPoints(argument, arg1, arg2, value1, value2):
    a = (value1 - value2)/(arg1 - arg2)
    b = value1 - arg1 * a

    return a*argument + b #point of linear fc


def zeroPadding(signal, finalLengthMultiply):
    signalLength = np.shape(signal)[0]
    toReturn = []
    for i in range(signalLength):
        toReturn.append(signal[i])

    for i in range(finalLengthMultiply):
        for k in range(signalLength):
            toReturn.append(0)

    return np.array(toReturn)


def functionValueInArg(args, fcArgs, fcVals):
    values = []

    for arg in args:

        firsArg = 0
        secArg = 0
        firstVal = 0
        secVal = 0
        argsCount = fcArgs.shape[0]
        for i in range(argsCount-1):
            gotNumbers = False
            if arg < fcArgs[0]:

                values.append(0)
                break;
            if i == argsCount-2: #arg biger than all but one fcArgs, -2 not to go outside list with i + 1
                # firsArg = fcArgs[i]
                # firstVal = fcVals[i]
                # secArg = fcArgs[i] + 1
                # secVal = fcVals[i] + 1
                # gotNumbers = True
                values.append(fcVals[i+1])

                break;
            if fcArgs[i] < arg < fcArgs[i + 1]: #arg is between two args of Fc
                firsArg = fcArgs[i]
                firstVal = fcVals[i]
                secArg = fcArgs[i + 1]
                secVal = fcVals[i + 1]
                gotNumbers = True

            if gotNumbers:
                values.append(valueBetweenPoints(arg, firsArg, secArg, firstVal, secVal))
                break;


    return np.array(values)


def generateExc(numberOfModes):
    # values in one row form one curve

    #setting where to save exc curves
    path_to_exc = config.ROOT_DIR + '/../eig/exc'

    path_to_omega_files = config.ROOT_DIR + '/../eig/omega'
    arguments = []
    excCurves = []
    f = []
    for mode in range(numberOfModes):
        f_v = rd.read_complex_omega(path_to_omega_files, mode)
        f.append(f_v)
        firstArg = f_v[0].real
        args = np.linspace(firstArg, firstArg + 5 * 1e5, 1000)
        arguments.append(args) # potrzebne?
        excTemp = []
        scale = np.random.random()
        for x in args:
            excTemp.append(np.exp(-scale * (x - firstArg)/1e4))

        excCurves.append(excTemp)

    excCurves = np.array(excCurves)
    arguments = np.array(arguments)
    rd.write_exc(arguments, excCurves, path_to_exc)


def getCurvesInSignalArgs(numberOfModes, args):
    path_to_exc = config.ROOT_DIR + '/../eig/exc'
    path_to_kvect_file = config.ROOT_DIR + '/../eig/kvect'
    path_to_omega_file = config.ROOT_DIR + '/../eig/omega'

    kvect = rd.read_kvect(path_to_kvect_file)
    omega = []
    for mode in range(numberOfModes):
        om = rd.read_complex_omega(path_to_omega_file, mode)
        omega.append(om.real)

    excArgs, excVal = rd.read_exc(path_to_exc)

    newKvect = []
    newAmps = []

    for mode in range(numberOfModes):
        newKvect.append(functionValueInArg(args, omega[mode], kvect))
        newAmps.append(functionValueInArg(args, excArgs[mode], excVal[mode]))
    newKvect = np.array(newKvect)
    newAmps = np.array(newAmps)
    return newKvect, newAmps


def getSignalSpectrum():
    samples = 1000
    time_end = 1e-3
    omega_max = 2.5 * 1e4 * 2 * np.pi
    time = np.linspace(0, time_end, samples)
    omega = np.linspace(0, omega_max, samples)

    fs = (samples / time_end)
    ################################################################
    freq_sampling = np.linspace(0, fs/2, int(samples/2 + 1) * 11 - 10)

    chirp = np.sin(omega*time)

    hanning = []
    for n in range(len(time)):
        hanning.append(0.5*(1-np.cos(np.pi*2*(n + 1)/len(time))))

    chirp_windowed = chirp # write 'chirp * hanning' to put window
    #############################
    chirp_windowed = zeroPadding(chirp_windowed, 10)
    time = np.linspace(0, time_end*11, samples*11)
    #############################
    f_chirp_windowed = np.fft.rfft(chirp_windowed)
    f_amp = np.sqrt(f_chirp_windowed.real**2 + f_chirp_windowed.imag**2)

    return chirp_windowed, time, freq_sampling * 2 * np.pi, f_chirp_windowed


def plotExcCurvesFromFile(numberOfModes):
    # function reads and plots curves

    path_to_exc = config.ROOT_DIR + '/../eig/exc'
    path_to_omega_files = config.ROOT_DIR + '/../eig/omega'

    args, exc = rd.read_exc(path_to_exc)

    plt.figure(1)

    for f, A in zip(args, exc):

        plt.plot(f, A)
        # plt.xlim([-5, 200])  # 600
        # plt.ylim([0.8, 1])  # 2000

    plt.show()


def plotTimeSignal(args, values):
    plt.figure(1)
    plt.plot(args, values)
        # plt.xlim([-5, 200])  # 600
        # plt.ylim([0.8, 1])  # 2000
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude [micro m]")
    plt.show()


def plotFrequencySignal(args, values):
    plt.figure(1)
    plt.plot(args/(1e3), np.sqrt(values.real**2 + values.imag**2))
    plt.xlim([-5, 60])  # 600
        # plt.ylim([0.8, 1])  # 2000
    plt.xlabel("Frequnecy [kHz]")
    plt.ylabel("Amplitude [micro m]")
    plt.show()


def plotCurves(args, values):
    values = np.array(values)
    modes = values.shape[0]
    plt.figure(1)
    argsToPlot = []
    valuesToPlot = []
    for i in range(modes):
        argsTemp = []
        valuesTemp = []
        appendFlag = False
        for arg, val in zip(args, values[i]):
            if val > 0:
                appendFlag = True
            if appendFlag:
                argsTemp.append(arg)
                valuesTemp.append(val)

        argsToPlot.append(argsTemp)
        valuesToPlot.append(valuesTemp)

    for i in range(modes):

        plt.plot(np.array(argsToPlot[i])/(1e3), np.array(valuesToPlot[i]))
        plt.xlim([-5, 60])  # 600
        plt.xlabel("Frequency [kHz]")
        plt.ylabel("Amplitude [-]")

    plt.show()


def lenghtPropagate(length, fChirp, kvect, excAmplitude):
    newFChirp = []

    kvect = kvect.transpose()
    excAmplitude = excAmplitude.transpose()
    ind = 1
    for amp, kv, excAmp in zip(fChirp, kvect, excAmplitude):

        resK = sum(kv)
        # resExc = sum(excAmp)
        resExc = excAmp[0] + excAmp[1] + excAmp[2] + excAmp[3]

        temp = amp * np.exp(-1j * resK * length *3e2) * resExc
        if(ind>100):
            temp *= 1.4
        newFChirp.append(temp)
        ind += 1


    return np.array(newFChirp)


def singAround(iterations, length, frequency, fChirp, kvect, excAmplitude, time):
    newFChirp = lenghtPropagate(length, fChirp, kvect, excAmplitude)
    for i in range(iterations):
        newFChirp = lenghtPropagate(length, newFChirp, kvect, excAmplitude)/10
        plotFrequencySignal(frequency, np.sqrt(newFChirp.real ** 2 + newFChirp.imag ** 2))
        plotTimeSignal(time, np.fft.irfft(newFChirp))
    return newFChirp


if __name__ == '__main__':
    numberOfModes = 8
    # generateExc(numberOfModes)
    chirp, time, omega, fChirp = getSignalSpectrum()

    # curves sampling
    kvect, excAmp = getCurvesInSignalArgs(numberOfModes, omega)

    fChirpPropagate = singAround(10, 2, omega / (2 * np.pi), fChirp, kvect, excAmp, time)
    plotTimeSignal(time, chirp)
    plotFrequencySignal(omega / (2 * np.pi), fChirp)
    plotFrequencySignal(omega / (2 * np.pi), fChirpPropagate)
    # plotSignal(frequncy, np.sqrt(fChirpPropagate.real**2 + fChirpPropagate.imag**2))
    # plotCurves(frequncy, kvect)
    plotCurves(omega / (2 * np.pi), excAmp)
