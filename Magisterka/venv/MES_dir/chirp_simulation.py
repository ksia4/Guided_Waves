import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# dane sygnalu wejsciowego
B = 50e3 # zakres częstotliwości w Hz
T = 1e-3 # czas trwania sygnału
samples = 1000 # liczba próbek

time = np.linspace(0, T, samples) # wektor czasu
frequency = np.linspace(0, samples/(2*T), samples/2 + 1)

#sygnal
chirp = np.array([np.sin((np.pi*B*t**2)/T) for t in time]) # testowany sygnał - chirp

