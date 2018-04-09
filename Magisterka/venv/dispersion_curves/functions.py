import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


# wypisuje elementy z tablicy, ktore sie powtarzaja
def multi_elements(some_list):
    for element1 in some_list:
        a = 0
        b = 1
        for element2 in some_list:
            if element1 == element2:
                a += 1
            if a > 1:
                a = 1
                b += 1
        if b > 1:
            print("Liczba wystapien:", b, "Element:", element1)


# sprawdza czy po sortowaniu listy wszystkie elementy dalej sie w niej zawieraja
def sorting_check(list1, list2):

    for element1 in list1:
        b1 = False
        for element2 in list2:
            if element1 == element2:
                b1 = True

        if b1 == False:
            print(element1)


# czytanie z plikow z danymi do macierzy mas i sztywnosci
def read_MARC_matrix(file_name, nnodes, nDOF):

    #konstrukcja macierzy zer o odpowiednim wymiarze
    matrix = np.zeros((nnodes*nDOF, nnodes*nDOF), dtype=float)
    #dzialania na plikach do odczytu
    with open(file_name) as file:
        for line in file:
            # linia bez wartosc
            if "DMIG" in line:
                ndC = int(line[-20:-17])
                nDOFC = int(line[-3:-1])
            # linia z wartoscia
            else:
                ndR = int(line[-36:-33])
                nDOFR = int(line[-20:-17])
                text = list(line[-17:-1])
                text[-4] = "e"
                val = float("".join(text))
                matrix[(ndR-1)*nDOF+nDOFR-1, (ndC-1)*nDOF+nDOFC-1] = val

    matrix = matrix + np.triu(matrix, 1).transpose()
    return matrix


# dane w pliku ze wspolrzednymi maja notacje naukowa bez literki 'e'
# funkcja ja dodaje w odpowiednie miejsce
def add_e_to_notation(str):

    list_str = list(str)

    #znajdowanie znaku przed ktorym trzeba wrzucic 'e'
    for ind, element in enumerate(list_str):
        if element == "+" or element == "-":
            ind1 = ind

    new_str = list_str[0:ind1]
    new_str.append("e")

    for i in range(len(list_str)-ind1):
        new_str.append(list_str[ind1+i])

    return "".join(new_str)


# odczytywanie wspolrzednych z pliku dat
def read_coordinates_from_DAT(file_name, nnodes, nDOF):

    #konstrukcja macierzy o odpowednim wymiarze
    matrix = np.zeros((nnodes, 4), dtype=float)

    read = False
    # False - no data in line
    # True - read data

    with open(file_name) as file:
        for ind, line in enumerate(file):

            # end of data - stop reading
            if "isotropic" in line:
                read = False

            # reading
            if read:

                if "0         1" in line:  # after coordinates, before data
                    continue

                data_ind = int(line[7:10])
                x = float(add_e_to_notation(line[10:30]))
                y = float(add_e_to_notation(line[30:50]))
                z = float(add_e_to_notation(line[50:70]))
                # print(x, "", y, "", z)

                matrix[ind - start_ind, 0] = data_ind - 1
                matrix[ind - start_ind, 1] = x
                matrix[ind - start_ind, 2] = y
                matrix[ind - start_ind, 3] = z

            # data starts in next line
            if "coordinates" in line:
                read = True
                start_ind = ind + 2
    return matrix


# sortowanie listy L i R tak, aby punkty plaszczyzn sobie odpowiadaly
def sort_coordinates(L, C, R, data_matrix):
    L1 = []
    R1 = []
    for index_c in C:
        temp_ly = []
        temp_lz = []
        temp_ry = []
        temp_rz = []

        for index_l in L:

           if abs(data_matrix[index_c, 2] - data_matrix[index_l, 2]) < 1e-10:
               temp_ly.append(int(data_matrix[index_l, 0]))

           if abs(data_matrix[index_c, 3] - data_matrix[index_l, 3]) < 1e-10:
               temp_lz.append(int(data_matrix[index_l, 0]))

        for i in temp_ly:
            for j in temp_lz:
                if i == j:
                    L1.append(j)

        for index_r in R:

           if abs(data_matrix[index_c, 2] - data_matrix[index_r, 2]) < 1e-10:
               temp_ry.append(int(data_matrix[index_r, 0]))

           if abs(data_matrix[index_c, 3] - data_matrix[index_r, 3]) < 1e-10:
               temp_rz.append(int(data_matrix[index_r, 0]))

        for i in temp_ry:
            for j in temp_rz:
                if i == j:
                    R1.append(j)

    return L1, R1


# sortowanie macierzy mas i sztywnosci zeby odpowiadaly indeksowaniu plaszczyzn
def sort_stiff_and_mass(M, K, L, C, R, ndof):

    all_planes = L + C + R
    M1 = np.zeros((np.shape(M)[0], np.shape(M)[0]))
    K1 = np.zeros((np.shape(K)[0], np.shape(K)[0]))

    for loop_ind, d_ind in enumerate(all_planes):
        M1[:, 3*loop_ind : 3*loop_ind + 3] = M[:, 3*d_ind : 3*d_ind + 3]
        K1[:, 3*loop_ind : 3*loop_ind + 3] = K[:, 3*d_ind : 3*d_ind + 3]

    M2 = np.zeros((np.shape(M)[0], np.shape(M)[0]))
    K2 = np.zeros((np.shape(K)[0], np.shape(K)[0]))

    for loop_ind, d_ind in enumerate(all_planes):
        M2[3*loop_ind : 3*loop_ind + 3, :] = M1[3*d_ind : 3*d_ind + 3, :]
        K2[3*loop_ind : 3*loop_ind + 3, :] = K1[3*d_ind : 3*d_ind + 3, :]

    temp = ndof*len(C)

    M0 = M2[temp : 2*temp, temp : 2*temp]
    KL = K2[temp : 2*temp, 0 : temp]
    K0 = K2[temp : 2*temp, temp : 2*temp]
    KR = K2[temp : 2*temp, 2*temp : 3*temp]

    return M0, KL, K0, KR


# indeksy dla punktow poszczegolnych plaszczyzn z d - punkty sobie odpowiadaja
def planes_indecies(data_matrix):
    L = []
    C = []
    R = []
    for ind, wiersz in enumerate(data_matrix):
        if wiersz[1]<10e-10:
            L.append(ind)
        if 1-10e-10 <wiersz[1]< 1+10e-10:
            C.append(ind)
        if 2-10e-10 < wiersz[1] < 2+10e-10:
            R.append(ind)
    L, R = sort_coordinates(L, C, R, data_matrix)
    return L, C, R


def find_eig(M0, KL, K0, KR):
    ux = 1
    # kvect = np.linspace(0, np.pi/2, num=51)
    kvect = np.linspace(0, np.pi / 8, num=51)
    Fsys = []

    for ind, k in enumerate(kvect):
        Ksys = K0 + np.exp(-1j * ux * k)*KL + np.exp(+1j * ux * k)*KR
        [F, V] = la.eig(Ksys, M0)
        F2 = np.sqrt(F)
        Fsys.append(F2.real)
        print("eig: ", ind+1, " z 51")
        print(np.shape(Fsys))

    return np.array(Fsys).transpose(), kvect


def calculate_dispersion_curves():
    ndof = 3
    nnodes = 724
    K = read_MARC_matrix("rod_v4_job1_glstif_0000", nnodes, ndof)
    M = read_MARC_matrix("rod_v4_job1_glmass_0000", nnodes, ndof)
    d = read_coordinates_from_DAT("rod_v4_job1.dat", nnodes, ndof)
    L, C, R = planes_indecies(d)
    # new_k = get_matrix_to_draw(K, L+C+R)
    # assembling.draw_matrix_sparsity(new_k)
    M0, KL, K0, KR = sort_stiff_and_mass(M, K, L, C, R, ndof)

    Fsys, kvect = find_eig(M0, KL, K0, KR)

    M0a = small_numbers_equal_zero(M0)
    K0a = small_numbers_equal_zero(K0)
    KLa = small_numbers_equal_zero(KL)
    KRa = small_numbers_equal_zero(KR)

    write_matrix_to_file("disp_k0", K0a)
    write_matrix_to_file("disp_kr", KRa)
    write_matrix_to_file("disp_kl", KLa)
    write_matrix_to_file("disp_m0", M0a)

    return Fsys, C, ndof, kvect


def draw_dispersion_curves(C, Fsys, kvect, ndof):
    plt.figure(1)
    plt.subplot(211)
    k_v = kvect*1e3
    E = 5000
    rho = 1000e-12

    # wszystkie krzywe
    # for ind in range(ndof*len(C)):
    #     f_v = Fsys[ind, :]/(2*np.pi)
    #     plt.plot(f_v*1e-3, k_v, 'g.', markersize=3)
    # plt.xlabel("Frequency [kHz]")
    # plt.ylabel("Wavenumber [rad/m]")
    # plt.xlim([0, 600])
    # plt.ylim([0, 2000])
    #
    # plt.subplot(212)
    # for ind in range(ndof*len(C)):
    #     f_v = Fsys[ind, :]/(2*np.pi)
    #     v_p = (2*np.pi*(f_v/k_v))/(np.sqrt(E/rho)*1e-3)
    #     plt.plot(f_v*1e-3, v_p, 'g.', markersize=3)
    # plt.xlabel("Frequency [kHz]")
    # plt.ylabel("Velocity [m/s]")
    # plt.xlim([0, 500])
    # plt.ylim([0, 50])
    # plt.show()

    # krzywe dla postaci o numerach w tablicy mode
    new_fsys = sort_columns(Fsys)

    modes = [i for i in range(50)]
    for ind in modes:
        f_v = new_fsys[ind, :]/(2*np.pi)
        plt.plot(f_v*1e-3, k_v, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Wavenumber [rad/m]")
    plt.xlim([0, 175])
    plt.ylim([0, 400])

    plt.subplot(212)
    for ind in modes:
        f_v = new_fsys[ind, :]/(2*np.pi)
        v_p = (2*np.pi*(f_v/k_v))/(np.sqrt(E/rho)*1e-3)
        plt.plot(f_v*1e-3, v_p, 'g.', markersize=3)
    plt.xlabel("Frequency [kHz]")
    plt.ylabel("Velocity [m/s]")
    plt.xlim([0, 175])
    plt.ylim([0, 60])
    plt.show()



def draw_plane(L, vertices):
    plt.scatter(vertices[L, 2], vertices[L, 3])
    plt.show()


# def write_matrix_to_file(filename, matrix):
#     with open(filename, "w") as file:
#         for i, row in enumerate(matrix):
#             for j, element in enumerate(row):
#                 to_save = str(i) + "  " + str(j) + "  " + str(element) + "  \n"
#                 file.write(to_save)
#             file.write("== \n")


def write_matrix_to_file(filename, matrix):

    with open(filename, "w") as file:
        for row in matrix:
            for element in row:
                # to_save = str(element) + " "
                # to_save.format(width=15)
                file.write("{0!s:16}".format(element))

            file.write(" \n")


def read_matrix_from_file(filename):
    matrix = []
    with open(filename, "r") as file:
        temp = []
        for line in file:
            if line[0:2] == "==":
                matrix.append(temp)
                temp = []
            else:
                elem = float(line[6:-1])
                temp.append(elem)
    return np.array(matrix)


def get_matrix_to_draw(matrix, indices):
    new_matrix = []
    for ind1 in indices:
        temp = []
        for ind2 in indices:
            temp.append(matrix[ind1, ind2])
        new_matrix.append(temp)
    return new_matrix


def small_numbers_equal_zero(matrix):
    new_matrix = []
    for row in matrix:
        temp = []
        for element in row:
            if - 1e-5 < element < 1e-8:
                temp.append(0)
            else:
                temp.append(element)
        new_matrix.append(temp)
    return np.array(new_matrix)

# modes test - sorting from highest frequencies
def sort_columns(matrix):
    new_matrix = []
    for i in range(len(matrix[0, :])):
        column = matrix[:, i]
        new_matrix.append(np.sort(column, 0))
    return np.array(new_matrix).transpose()

# d = read_coordinates_from_DAT("rod_v4_job1.dat", 724, 3)
# print("dsad")