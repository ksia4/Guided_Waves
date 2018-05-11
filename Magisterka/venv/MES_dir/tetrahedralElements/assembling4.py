import numpy as np
import matplotlib.pyplot as plt
from MES_dir.tetrahedralElements import calculations4, gauss4


def assemble_global_stiff_matrix(vertices, indices, young_modulus, poisson_coefficient):
    print("Obliczanie macierzy sztywności")
    global_matrix = np.zeros([np.shape(vertices)[0]*3, np.shape(vertices)[0]*3])
    # print("global ", np.shape(global_matrix))
    for elem_ind in indices:
        n = calculations4.shape_functions(vertices, elem_ind)
        ki = calculations4.stiff_local_matrix(n, vertices, elem_ind, young_modulus, poisson_coefficient)

        for i in range(len(elem_ind)):  #petla po liczbie punktow w jednej osi
            for j in range(len(elem_ind)):  # w drugiej osi
                for m in range(3):  # po wspolrzednych
                    for n in range(3):  # po wspolrzednych

                        global_matrix[elem_ind[i]*3 + m, elem_ind[j]*3 + n] += ki[i*3 + m, j*3 + n]
    return np.array(global_matrix)


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

    global_matrix = np.zeros([np.shape(vertices)[0] * 3, np.shape(vertices)[0] * 3])
    mass_neutral = calculations4.mass_local_matrix(density)

    sfn = gauss4.shape_functions_natural()


    for elem_ind in indices:

        j = gauss4.jacobian(vertices[elem_ind], sfn)
        mass = mass_neutral*j

        for i in range(len(elem_ind)):  #petla po liczbie punktow w jednej osi
            for j in range(len(elem_ind)):  # w drugiej osi
                for m in range(3):  # po wspolrzednych
                    for n in range(3):  # po wspolrzednych

                        global_matrix[elem_ind[i]*3 + m, elem_ind[j]*3 + n] += mass[i*3 + m, j*3 + n]
    return np.array(global_matrix)


def focuse_matrix_rows(matrix):
    new_matrix = np.zeros(np.shape(matrix))

    for ind, row in enumerate(matrix):
        sum = 0
        for element in row:
            sum += element
        new_matrix[ind, ind] = sum
    return new_matrix
