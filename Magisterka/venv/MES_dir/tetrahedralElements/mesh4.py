import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spt
import numpy.linalg as la



def circle_plane_verticies(x, radius, number_of_points):
    angles = np.array(np.linspace(0, 2*np.pi, number_of_points))
    angles1 = np.delete(angles, -1, 0)
    vertices = []
    for fi in angles1:
        y = radius*np.cos(fi)
        z = radius*np.sin(fi)
        vertex = [x, y, z]
        vertices.append(vertex)
    return np.array(vertices)


def circle_mesh_full(length, radius, number_of_circles, number_of_points):
    vertices = []
    circles_rad = np.linspace(radius/number_of_circles, radius, number_of_circles)
    for i in range(length + 1):
        vertices.append([i, 0, 0])
        for j, r in enumerate(circles_rad):

            circle = circle_plane_verticies(i, r, (number_of_points+1) * (j+1))
            for row in circle:
                vertices.append(row)
    return np.array(vertices)


def circle_mesh_full2(length, radius, number_of_circles, number_of_points):
    vertices = []
    circles_rad = np.linspace(radius/number_of_circles, radius, number_of_circles)
    for i in range(length):
        vertices.append([i- i*0.01, 0, 0])
        for j, r in enumerate(circles_rad):

            circle = circle_plane_verticies(i-i*0.01, r, (number_of_points+1) * (j+1))
            for row in circle:
                vertices.append(row)
    return np.array(vertices)


def circle_mesh_sparse(length, radius, number_of_circles, number_of_points):
    vertices = [] #central point
    circles_rad = np.linspace(radius/number_of_circles, radius, number_of_circles)
    for i in range(length):
        vertices.append([i, 0, 0])
        for r in circles_rad:
            circle = circle_plane_verticies(i, r, number_of_points + 1)
            for row in circle:
                vertices.append(row)
    return np.array(vertices)


def draw_plane(vertices):
    plt.scatter(vertices[:, 1], vertices[:, 2])
    plt.show()


def draw_bar(vertices):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.set_xlim([-10, 10])
    # ax.set_zlim([-200, 200])
    plt.show()


def correct_volume_sign(vertices, indices):
    corrected_indices = []

    for row in indices:
        vert = vertices[row, :]

        temp1 = [1]
        temp1.extend(list(vert[0]))
        temp2 = [1]
        temp2.extend(list(vert[1]))
        temp3 = [1]
        temp3.extend(list(vert[2]))
        temp4 = [1]
        temp4.extend(list(vert[3]))

        a = [temp1, temp2, temp3, temp4]
        a1 = np.array(a)

        volume = la.det(a1)
        if volume > 0:
            corrected_indices.append(row)
        else:
            new_row = [row[1], row[0], row[2], row[3]] # zmiana dwoch elementow zmieni znak objetosci
            corrected_indices.append(new_row)

    return corrected_indices


def triangulation(vertices):
    tri = spt.Delaunay(vertices)
    indices = tri.simplices.copy()

    # wyszukiwanie elementow z wierzcholkami wspolplaszczyznowymi
    planar = []
    for ind, tetr in enumerate(indices):

        determinant = 0
        matrix = []
        for elem in tetr:
            l = [1, vertices[elem, 0], vertices[elem, 1], vertices[elem, 2]]
            matrix.append(l)
        matrix1 = np.array(matrix)
        # print(la.det(matrix1))
        if abs(la.det(matrix1)) < 1e-10:
            planar.append(ind)
    # usuwanie wierszy z wierzcholkami wspolplaszczyznowymi
    indices1 = np.delete(indices, planar, 0)

    indices2 = correct_volume_sign(vertices, indices1)

    return np.array(indices2)


def draw_triangulation(vertices, indices):
    fig = plt.figure()
    lines = []
    for i, ind in enumerate(indices):
        for i in ind:
            for j in ind:
                if j != i and not [i, j] in lines and not [j, i] in lines:
                    lines.append([i, j])
    ax = fig.gca(projection='3d')
    # ax.set_xlim([-1, 2])
    # ax.triplot(np.array(points[:, 0]), np.array(points[:, 1]), np.array(points[:, 2]), tri.simplices.copy()))
    # ax.plot(np.array(points)[:, 0], np.array(points)[:, 1], np.array(points)[:, 2], 'o')
    for line in lines:
        ax.plot(vertices[line[0: 2], 0], vertices[line[0: 2], 1], vertices[line[0: 2], 2], 'r-')

    ax.scatter(vertices[indices[0:3, :], 0], vertices[indices[0:3, :], 1], vertices[indices[0:3, :], 2])
    plt.show()


# def mesh_fc():
#     length = 4000
#     planes = 10
#     radius = 20
#     density = 10
#     number_of_edge_points = 15
#     verticies = equal_size_mesh(length, planes, radius, density, number_of_edge_points)
#     return verticies








