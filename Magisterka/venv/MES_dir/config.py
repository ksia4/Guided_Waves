from numpy import pi
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

ndof = 3    # liczba stopni swobody

kvect_min = 0
kvect_no_of_points = 0
kvect_max = 2*pi

k = []  # globalna macierz sztywnosci
m = []  # globalna macierz mas
m_focused_rows = [] #globalna macierz mas - skupiona wierszami
ml = []
m0 = []
mr = []
kl = []
k0 = []
kr = []

force = []


# Stale materialowe
young_mod = 70000 #70000
poisson_coef = 0.3 #0.33333
density = 2.7*1e-9 # 2.7*1e-9


# Wykresy

# Wyswietlanie siatki na plaszczyznie, siatki w 3D i triangulacji
show_plane = False
show_bar = False
show_elements = False