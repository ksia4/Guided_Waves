import numpy as np
import matplotlib.pyplot as plt
from MES_dir import mode_sampling
from MES_dir import readData as rd


# dane sygnalu wejsciowego
B = 50e3 # zakres częstotliwości w Hz
T = 1e-3 # czas trwania sygnału
samples = 1000 # liczba próbek

time = np.linspace(0, T, samples) # wektor czasu
frequency = np.linspace(0, samples/(2*T), samples/2 + 1)

#sygnal
chirp = np.array([np.sin((np.pi*B*t**2)/T) for t in time]) # testowany sygnał - chirp

f_chirp = np.fft.rfft(chirp) # transformata Furiera z chirp-a
timestamp = T/samples
ff = np.fft.fftfreq(chirp.size, d=timestamp)
x = 2000 # odleglosc pomiaru

path1 = "../eig/omega"
path2 = "../eig/kvect"

temp_omega = []
for i in range(4):
    temp_omega.append(rd.read_complex_omega(path1, i))

omega = np.array(temp_omega)
kvect = rd.read_kvect(path2)

# frequencies = np.linspace(min(real_frequency[0]) + 1e-10, min(real_frequency[0]) + 100e3, samples)
temp = []
print("frequency length ", len(frequency))
k = mode_sampling.curve_sampling(omega[0], kvect, frequency)
print("k length: ", len(k))
for ch, k in zip(f_chirp, k):
    temp.append(ch*np.exp(1j*k*x))
transformed_chirp = np.array(temp)

result_chirp = np.fft.irfft(transformed_chirp)

plt.figure(1)

plt.subplot(321)
plt.plot(time, chirp)
plt.title('Chirp')

plt.subplot(322)
plt.plot(frequency * 1e-3, f_chirp.real)
plt.title('Chirp fft (real)')

plt.subplot(324)
plt.plot(frequency * 1e-3, transformed_chirp)
plt.title('Chirp fft after propagating {}[m] (real)'.format(x/1000))

plt.subplot(323)
plt.plot(time, result_chirp)
plt.title('Chirp after propagating {}[m]'.format(x/1000))

plt.subplot(325)
plt.plot(time, result_chirp-chirp)
plt.title('Output - Input')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5,
                    wspace=0.35)

plt.show()
