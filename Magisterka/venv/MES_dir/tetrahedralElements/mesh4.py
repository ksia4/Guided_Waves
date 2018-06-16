import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spt
import numpy.linalg as la

def circlePlaneVerticies(x, radius, numberOfPoints):
    angles = np.array(np.linspace(0, 2*np.pi, numberOfPoints))
    angles1 = np.delete(angles, -1, 0)
    vertices = []
    for fi in angles1:
        y = radius*np.cos(fi)
        z = radius*np.sin(fi)
        vertex = [x, y, z]
        vertices.append(vertex)
    return np.array(vertices)

#Siatka o zadanym promieniu - radius. Tworzy punkt środkowy i okregi w ilości - circles.
#First circle oznacza ilość punktów na pierwszym okregu.
#Add nodes oznacza ile punktów więcej będzie na każdym kolejnym okręgu.
#Zwraca macierz n x 3, gdzie n to liczba węzłów. W kolumnach są współrzędne punktów.
def circleMeshFull(radius, numberOfCircles, numberOfPoints):
    vertices = []
    circles_rad = np.linspace(radius / numberOfCircles, radius, numberOfCircles)
    for i in range(3):
        vertices.append([i, 0, 0])
        for j, r in enumerate(circles_rad):

            circle = circlePlaneVerticies(i, r, (numberOfPoints + 1) * (j + 1))
            for row in circle:
                vertices.append(row)
    return np.array(vertices)

#Siatka o zadanym promieniu - radius. Tworzy punkt środkowy i okregi w ilości - circles.
#First circle oznacza ilość punktów na pierwszym okregu.
#Na każdym okręgu jest tyle samo punktów
#Zwraca macierz n x 3, gdzie n to liczba węzłów. W kolumnach są współrzędne punktów.
def circleMeshSparse(radius, numberOfCircles, numberOfPoints):
    vertices = [] #central point
    circles_rad = np.linspace(radius/numberOfCircles, radius, numberOfCircles)
    for i in range(3):
        vertices.append([i, 0, 0])
        for r in circles_rad:
            circle = circlePlaneVerticies(i, r, numberOfPoints + 1)
            for row in circle:
                vertices.append(row)
    return np.array(vertices)

#Zmienia kolejność węzłów jeśli objętość liczona przy pomocy wyznacznika jest ujemna.
def correctVolumeSign(vertices, indices):
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

#Tworzy elementy czworościenne wykorzystując przestrzenną triangulację Delaunay'a.
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

    indices2 = correctVolumeSign(vertices, indices1)

    return np.array(indices2)

#Rysuje węły na płaszczyźnie.
def drawPlane(vertices):
    plt.scatter(vertices[:, 1], vertices[:, 2])
    plt.show()

#Rysuje wszystkie płaszczyzn w 3D.
def drawBar(vertices):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.set_xlim([-10, 10])
    # ax.set_zlim([-200, 200])
    plt.show()

#Rysje układ elementów czworościennych w 3D.
def drawTriangulation(vertices, indices):
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

if __name__ == "__main__":
    vertices = circleMeshFull(10, 10, 10)
    drawPlane(vertices)
    indices = triangulation(vertices)
    drawBar(vertices)
    drawTriangulation(vertices, indices)







