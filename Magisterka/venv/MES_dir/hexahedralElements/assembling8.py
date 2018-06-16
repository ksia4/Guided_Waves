import numpy as np
import matplotlib.pyplot as plt
from MES_dir.hexahedralElements import calculations8
import time


def assemble_global_stiff_matrix(vertices, indices):
    print("Obliczanie macierzy sztywności")
    global_matrix = np.zeros([np.shape(vertices)[0]*3, np.shape(vertices)[0]*3])
    # print("global ", np.shape(global_matrix))
    for elem_ind in indices:
        start = time.clock()
        ki = calculations8.localStiffMatrix(vertices[elem_ind])
        print("Czas obliczania macierzy sztywności jednego elementu: ", time.clock() - start)

        for i in range(len(elem_ind)):  #petla po liczbie punktow w jednej osi
            for j in range(len(elem_ind)):  # w drugiej osi
                for m in range(3):  # po wspolrzednych
                    for n in range(3):  # po wspolrzednych

                        global_matrix[elem_ind[i]*3 + m, elem_ind[j]*3 + n] += ki[i*3 + m, j*3 + n]
    return np.array(global_matrix)

#Pozwala zaobserwować rzadkość macierzy.
def draw_matrix_sparsity(matrix):
    max = 0
    min = 0

    for row in matrix:
        for element in row:
            if element < min:
                min = element

    matrix1 = []
    for row in matrix:
        temp1 = []
        for element in row:
            temp1.append(element + abs(min))
        matrix1.append(temp1)

    for row in matrix1:
        for element in row:
            if element > max:
                max = element

    for row in matrix1:
        for element in row:
            element = element/max
    # plt.imshow(matrix, ("nazwa", 2))
    plt.imshow(matrix, cmap='binary')
    plt.show()

def assemble_global_mass_matrix(vertices, indices, density):
    print("Obliczanie macierzy mas")
    global_matrix = np.zeros([np.shape(vertices)[0]*3, np.shape(vertices)[0]*3])
    # print("global ", np.shape(global_matrix))
    for elem_ind in indices:
        start = time.clock()
        mass = calculations8.localMassMatrix(vertices[elem_ind])
        print("Czas obliczania macierzy mas jednego elementu:  ", time.clock() - start)

        for i in range(len(elem_ind)):  #petla po liczbie punktow w jednej osi
            for j in range(len(elem_ind)):  # w drugiej osi
                for m in range(3):  # po wspolrzednych
                    for n in range(3):  # po wspolrzednych

                        global_matrix[elem_ind[i]*3 + m, elem_ind[j]*3 + n] += mass[i*3 + m, j*3 + n]
    return np.array(global_matrix)

#Wyznacza macierz skupioną.
def focuse_matrix_rows(matrix):
    new_matrix = np.zeros(np.shape(matrix))

    for ind, row in enumerate(matrix):
        sum = 0
        for element in row:
            sum += element
        new_matrix[ind, ind] = sum
    return new_matrix
