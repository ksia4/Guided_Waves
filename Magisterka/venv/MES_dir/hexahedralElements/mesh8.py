import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial as spt
import numpy.linalg as la

def plotTrianglesToCheck(vertices, triangleIndices, triangles):
    verticesYZ = vertices[:, 1:3]
    print(triangleIndices[0])
    for tri in triangles:
        if not (tri > 0.01 and tri < 0.9):
            plt.plot([verticesYZ[int(triangleIndices[tri, 0]), 0], verticesYZ[int(triangleIndices[tri, 1]), 0]], [verticesYZ[int(triangleIndices[tri, 0]), 1], verticesYZ[int(triangleIndices[tri, 1]), 1]])
            plt.plot([verticesYZ[int(triangleIndices[tri, 0]), 0], verticesYZ[int(triangleIndices[tri, 2]), 0]], [verticesYZ[int(triangleIndices[tri, 0]), 1], verticesYZ[int(triangleIndices[tri, 2]), 1]])
            plt.plot([verticesYZ[int(triangleIndices[tri, 1]), 0], verticesYZ[int(triangleIndices[tri, 2]), 0]], [verticesYZ[int(triangleIndices[tri, 1]), 1], verticesYZ[int(triangleIndices[tri, 2]), 1]])
    plt.show()

def circleMeshFull(length, numberOfPlanes, radius, firstCircle, addNodes, circles):

    def plane(x, radius, firstCircle, addNodes, circles):
        vertices = []
        vertices.append([x, 0, 0])

        circleRadius_temp = np.linspace(0, radius, circles + 1)
        circleRadius = np.delete(circleRadius_temp, 0)

        nodesOnCircle = firstCircle

        for ind, r in enumerate(circleRadius):
            angle_temp = np.linspace(0, 2*np.pi, nodesOnCircle + 1)
            angle = np.delete(angle_temp, -1, 0)
            for fi in angle:
                y = r*np.cos(fi)
                z = r*np.sin(fi)

                vertices.append([x, y, z])

            nodesOnCircle += addNodes

        return vertices

    planeX = np.linspace(0, length, numberOfPlanes)

    vertices = []

    for px in planeX:
        planeVertices = plane(px, radius, firstCircle, addNodes, circles)
        for vertex in planeVertices:
            vertices.append(vertex)

    numberOfPointsOnLastCircle = firstCircle + addNodes * (circles - 1)

    return np.array(vertices), numberOfPointsOnLastCircle #węzły siatki i liczba punktow na najwiekszym okregu

# Do porównania z wynikami od doktora
def circleMeshSparse(length, numberOfPlanes, radius, firstCircle, circles):
    # first circle - 17, circles - 10
    def plane(x, radius, firstCircle, circles):
        vertices = []
        vertices.append([x, 0, 0])

        circleRadius_temp = np.linspace(0, radius, circles + 1)
        circleRadius = np.delete(circleRadius_temp, 0)

        nodesOnCircle = firstCircle

        for ind, r in enumerate(circleRadius):
            angle_temp = np.linspace(0, 2*np.pi, nodesOnCircle + 1)
            angle = np.delete(angle_temp, -1, 0)
            for fi in angle:
                y = r*np.cos(fi)
                z = r*np.sin(fi)

                vertices.append([x, y, z])

        return vertices

    planeX = np.linspace(0, length, numberOfPlanes)

    vertices = []

    for px in planeX:
        planeVertices = plane(px, radius, firstCircle, circles)
        for vertex in planeVertices:
            vertices.append(vertex)

    numberOfPointsOnLastCircle = firstCircle

    return np.array(vertices), numberOfPointsOnLastCircle #węzły siatki


def createFiniteElements(vertices, pointsOnLastCircle, length, numberOfPlanes):

    def triangulation1():
        # Triangulacja - dobieranie punktów dla elementów skończonych na pierwszej płaszczyźie
        # print(vertices[:, 1:3])
        firstPlaneVerticesYZ = vertices[0: int(len(vertices[:, 0])/3), 1:3]
        tri = spt.Delaunay(firstPlaneVerticesYZ)
        return vertices, tri.simplices.copy()

    # def sortIndices(indices):
    #     rowSortedIndices = []
    #     for row in indices:
    #         rowSortedIndices.append(np.sort(row))
    #     rowSortedIndices = np.array(rowSortedIndices)
    #     # print(rowSortedIndices)
    #     columnSortedIndices = rowSortedIndices[np.argsort(rowSortedIndices[:, 0])]
    #     return np.array(columnSortedIndices)

    def sortIndices(indices):
        rowSortedIndices = np.sort(indices)
        columnSortedIndices = rowSortedIndices[np.lexsort((rowSortedIndices[:, 1], rowSortedIndices[:, 0]))]
        return np.array(columnSortedIndices)

    # lepiej tej funkcji nawet nie czytać xD
    def joinTriangles(vertices, sortedIndices):

        def sortTetragonIndices(verticesYZ, indices):

            def getTetragonCentre(elementVerticesYZ):
                y = 0
                z = 0

                for point in elementVerticesYZ:
                    y += point[0]
                    z += point[1]
                return [y / 4, z / 4]

            def getNodeToCentreAngle(elementVerticesYZ, elementCentralPoint, elementIndices):
                angles = [[], []]
                for elementPoint, pointIndex in zip(elementVerticesYZ, elementIndices):
                    y = elementCentralPoint[0] - elementPoint[0]
                    z = elementCentralPoint[1] - elementPoint[1]
                    angle = np.arctan2(z, y) * 180 / np.pi
                    if angle < 0:
                        angle += 360
                    angles[0].append(angle)
                    angles[1].append(pointIndex)
                return angles

            def arrangeElementIndices(elementVerticesYZ, elementIndices):

                # znajduje średnią ze współrzędnych Y i Z punktów
                elemCentre = getTetragonCentre(elementVerticesYZ)

                # znajduje kąt pomiędzy prostą punkt - centralny punkt, dla każdego punktu
                angles = getNodeToCentreAngle(elementVerticesYZ, elemCentre, elementIndices)

                # zapisanie jako tablica numpy żeby sortowanie działało
                # niestety z intów robią się tu floaty
                angles = np.array(angles)

                # ustawia indeksy na podstawie rosnacego kąta
                angles = angles[:, np.argsort(angles[0, :], axis=0)]
                return [int(angles[1, 0]), int(angles[1, 1]), int(angles[1, 2]), int(angles[1, 3])]  # indices

            sortedTetragonIndices = []

            # dla każdego czworokata sortuje indeksy przeciwnie do wskazowek zegara
            # zapewnia to, że pole elementu liczone wyznacznikiem wyjdzie dodatnie

            # elementuIndices - indeksy punktow elementu w macierzy verticesYZ
            for elementIndices in indices:
                elementVerticesYZ = []
                # znajduje wpolrzedne punktow dla elementu
                for index in elementIndices:
                    elementVerticesYZ.append(verticesYZ[index])

                sortedElementIndices = arrangeElementIndices(elementVerticesYZ, elementIndices)

                sortedTetragonIndices.append(sortedElementIndices)

            return sortedTetragonIndices

        verticesXY = vertices[:, 1:3]

        tetragonIndices = []

        # zbiór numerow trojkatow juz przydzielonych - 1e-1 jest bo się nie da pustego zbioru stworzyć
        trianglesChecked = {1e-1}

        # dla każdego trojkata szukamy drugiego przylegajacego i laczymy w czworokat (na indeksach)

        for firstInd, firstTriangle in enumerate(sortedIndices):

            if firstInd not in trianglesChecked:

                # dodajemy numer trojkąta sprawdzanego
                trianglesChecked.add(firstInd)

                possibleToJoin = []
                secIndList = []

                for secInd, secondTriangle in enumerate(sortedIndices):
                    if secInd not in trianglesChecked:
                        # jeśli mają wspolny bok to stworz na idenksach czworokąt
                        if secondTriangle[0] in firstTriangle and secondTriangle[1] in firstTriangle:
                            possibleToJoin.append(secondTriangle)
                            secIndList.append(secInd)
                            break

                        if secondTriangle[0] in firstTriangle and secondTriangle[2] in firstTriangle:
                            possibleToJoin.append(secondTriangle)
                            secIndList.append(secInd)
                            break

                        if secondTriangle[1] in firstTriangle and secondTriangle[2] in firstTriangle:
                            possibleToJoin.append(secondTriangle)
                            secIndList.append(secInd)
                            break

                possibleToJoin = np.array(possibleToJoin) # pierwszy jaki znajdzie możliwy do połączenia
                secIndList = np.array(secIndList)

                if np.shape(possibleToJoin)[0] != 0:
                    wholeInOne = []
                    for triangle, secInd in zip(possibleToJoin, secIndList):
                        temp = []
                        for index in triangle:
                            temp.append(index)
                        temp.append(secInd)
                        wholeInOne.append(temp)
                    wholeInOne = np.array(wholeInOne)

                    wholeInOne = wholeInOne[np.lexsort((wholeInOne[:, 2], wholeInOne[:, 1], wholeInOne[:, 0]))]

                    if wholeInOne[0, 0] in firstTriangle and wholeInOne[0, 1] in firstTriangle:
                        tetragonIndices.append(
                            [firstTriangle[0], firstTriangle[1], firstTriangle[2], wholeInOne[0, 2]])
                        trianglesChecked.add(wholeInOne[0, 3])
                        # plotTrianglesToCheck(vertices, sortedIndices, trianglesChecked)

                    if wholeInOne[0, 0] in firstTriangle and wholeInOne[0, 2] in firstTriangle:
                        tetragonIndices.append(
                            [firstTriangle[0], firstTriangle[1], firstTriangle[2], wholeInOne[0, 1]])
                        trianglesChecked.add(wholeInOne[0, 3])
                        # plotTrianglesToCheck(vertices, sortedIndices, trianglesChecked)

                    if wholeInOne[0, 1] in firstTriangle and wholeInOne[0, 2] in firstTriangle:
                        tetragonIndices.append(
                            [firstTriangle[0], firstTriangle[1], firstTriangle[2], wholeInOne[0, 0]])
                        trianglesChecked.add(wholeInOne[0, 3])
                        # plotTrianglesToCheck(vertices, sortedIndices, trianglesChecked)

        tetragonIndices = sortTetragonIndices(verticesXY, tetragonIndices)
        return np.array(tetragonIndices)

    def fillHolesInBar(tetragonIndices):
        def indicesToCover():
            numberOfPointsForPlane = len(vertices[:, 0])/3 #liczba wierszy/3

            #indeksy punktow na najwiekszym okręgu - tam są dziury!
            #indeksowanie od 0
            indicesOnLastCircle = [numberOfPointsForPlane - i - 1 for i in range(pointsOnLastCircle)]

            indicesNOTToCoverWithElement = []

            for i in range(len(indicesOnLastCircle)):
                for tetragon in tetragonsIndices:
                    if indicesOnLastCircle[i] in tetragon and indicesOnLastCircle[i-1] in tetragon:
                        if indicesOnLastCircle[i] not in indicesNOTToCoverWithElement:
                            indicesNOTToCoverWithElement.append(indicesOnLastCircle[i])

            indicesToCoverWithElement = []
            for ind in indicesOnLastCircle:
                if ind not in indicesNOTToCoverWithElement:
                    indicesToCoverWithElement.append(ind)
            return indicesToCoverWithElement
            # return [i + 1 for i in reversed(indicesToCoverWithElement)] # jak się doda 1 to działa

        def addNewVertices(indicesToCover):
            def findAngle(currentIndex, angleBetweenPoints):
                cVertex = vertices[int(currentIndex+1), 1:3] #YZ
                cAngle = np.arctan2(cVertex[1], cVertex[0])
                # Wynik jest z przedziału -pi do pi a chcę 0 do 2pi
                if cAngle < 0:
                    cAngle += np.pi * 2

                return cAngle - angleBetweenPoints/2 # kąt nowego punktu

            def findRadius(currentIndex):
                cVertex = vertices[int(currentIndex), 1:3] #YZ
                return np.sqrt(cVertex[0]**2 + cVertex[1]**2)

            angleBetweenPoints = 2 * np.pi / pointsOnLastCircle

            #dodajemy punkty na ostatnim okręgu dla wszystkich płaszczyzn
            newVerticesTemp = []
            xCoor = np.linspace(0, length, numberOfPlanes)
            for x in xCoor:
                for i in range(len(indicesToCover)):
                    angle = findAngle(indicesToCover[i], angleBetweenPoints)
                    radius = findRadius(indicesToCover[i])

                    newVertex = [x, radius * np.cos(angle), radius * np.sin(angle)]
                    newVerticesTemp.append(newVertex)
                    newVertices = np.array(newVerticesTemp)

            numberOfNewVertOnPlane = int(len(newVertices)/numberOfPlanes)
            numberOfOldVerticesOnPlane = int(len(vertices)/numberOfPlanes)
            newVerticesArray = []

            for i in range(numberOfPlanes):
                for oldVert in vertices[i * numberOfOldVerticesOnPlane: (i + 1) * numberOfOldVerticesOnPlane, :]:
                    # print("old ", oldVert)
                    newVerticesArray.append(oldVert)
                for newVert in newVertices[i * numberOfNewVertOnPlane: (i + 1) * numberOfNewVertOnPlane, :]:
                    # print("new ", newVert)
                    newVerticesArray.append(newVert)

            return np.array(newVerticesArray)

        def addNewTetragons(indicesToCover):

            def findForthPoint(firstIndex, thirdIndex):
                firstPointTetra = []
                thirdPointTetra = []
                # czworokaty zawierające pierwszy indeks nowego elementu
                for indices in tetragonIndices:
                    for index in indices:
                        if int(firstIndex) == int(index):
                            firstPointTetra.append(indices)
                            break
                # czworokąty zawierające trzeci indeks nowego elementu
                for indices in tetragonIndices:
                    for index in indices:
                        if int(thirdIndex) == int(index):
                            thirdPointTetra.append(indices)
                            break

                # indeks wspólny dla elementu z pierwszym i trzecim indeksem
                for firstTetra, thirdTetra in zip(firstPointTetra, thirdPointTetra):
                    for first in firstTetra:
                        for third in thirdTetra:
                            if int(first) == int(third):
                                if first != firstIndex and first != thirdIndex:
                                    return first

                return -1 # jeśli nie znajdzie

            newTetragonsIndicesArray = []
            for tetragon in tetragonIndices:
                newTetragonsIndicesArray.append(tetragon)

            numberOfPointsForPlane = len(vertices[:, 0]) / 3  # liczba wierszy/3
            indicesOnLastCircleTemp = [numberOfPointsForPlane - i - 1 for i in range(pointsOnLastCircle)]
            indicesOnLastCircle = [i for i in reversed(indicesOnLastCircleTemp)]

            for ind, indToCover in enumerate(indicesToCover):
                newTetragonIndices = []
                #pierwszy punkt jest za nowym punktem zgodnie z ruchem wskazowek zegara
                for i in range(len(indicesOnLastCircle)):

                    if int(indicesOnLastCircle[i]) == int(indToCover):

                        newTetragonIndices.append(indicesOnLastCircle[i])


                #drugi punkt jest nowym punktem, w indeksach są zaraz za starymi dla każdej płaszczyzny
                newTetragonIndices.append(numberOfPointsForPlane + ind)

                #trzeci punkt jest przed nowym punktem zgodnie z ruchem wskazówek zegara
                newTetragonIndices.append(indToCover + 1)

                #czwarty punkt jest w dwóch elementach, z których jeden zawiera indeks punktu przed a drugi za nowym pkt.
                newTetragonIndices.append(findForthPoint(newTetragonIndices[0], newTetragonIndices[2]))

                newTetragonsIndicesArray.append(newTetragonIndices)

            # for newTetragon in newTetragonsIndicesArray:
            #
            #
            return newTetragonsIndicesArray

        indToCover = indicesToCover()
        print("ind to cover", len(indToCover))
        if len(indToCover) != 0:
            newVertices = addNewVertices(indToCover)

            newTetragonsIndices = addNewTetragons(indToCover)

            return np.array(newVertices), np.array(newTetragonsIndices)
        return vertices, tetragonsIndices

    def createHexahedrons(tetragonsIndices, nodesOnPlane):
        print("hexa: nodesOnPlane ", nodesOnPlane)
        hexahedronsIndices = []
        for indices in tetragonsIndices:
            newIndices1 = []
            for index in indices:
                newIndices1.append(index)
            for index in indices:
                newIndices1.append(index + nodesOnPlane)
            hexahedronsIndices.append(newIndices1)

        for indices in tetragonsIndices:
            newIndices2 = []
            for index in indices:
                newIndices2.append(index + nodesOnPlane)
            for index in indices:
                newIndices2.append(index + 2 * nodesOnPlane)
            hexahedronsIndices.append(newIndices2)

        return np.array(hexahedronsIndices)

    # triangluacja pierwszej plaszczyzny
    vertOnPlane, indicesOnPlane = triangulation1()

    # sortowanie przed łączeniem
    sortedIndices = sortIndices(indicesOnPlane)

    # łączenie trójkątów w czworokąty
    tetragonsIndices = joinTriangles(vertOnPlane, sortedIndices)
    print("liczba czworokatow ", len(tetragonsIndices[:, 0]))
    newVertices, newTetragonsIndices = fillHolesInBar(tetragonsIndices)
    print("nowe pkt ", len(newVertices[:, 0]))
    print("stare czworokąty ", len(tetragonsIndices[:, 0]))
    print("nowe czworokąty ", len(newTetragonsIndices[:, 0]))
    # tworzenie bryl
    nodesOnPlane = int(len(newVertices[:, 0])/3)
    hexaherdonsIndices = createHexahedrons(newTetragonsIndices, nodesOnPlane)
    # nodesOnPlane = int(len(vertices[:, 0])/3)
    # hexaherdonsIndices = createHexahedrons(tetragonsIndices, nodesOnPlane)
    print("Liczba węzłów: ", len(vertices[:, 0]))
    print("Liczba elementów skończonych: ", len(hexaherdonsIndices[:, 0]))
    # print(vertices)
    # print(np.array(hexaherdonsIndices))
    return np.array(newVertices), np.array(hexaherdonsIndices).astype(int)

def brickMesh(length, numberOfPlanes, radius, circles, pointsOnCircle):
    def planeVertices():
        #radius of circles
        rad_temp = np.linspace(0, radius, circles + 1)
        rad = np.delete(rad_temp, 0)
        #angles
        angle_temp = np.linspace(0, np.pi*2, pointsOnCircle + 1)
        angle = np.delete(angle_temp, -1)

        vertices_plane = []
        #central point
        vertices_plane.append([0, 0, 0])
        #first circle
        j = 8
        for i in range(j):
            # fi = 0 + i*np.pi/(j/2)
            fi = np.pi/j + i*np.pi/(j/2)
            vertices_plane.append([0, 0.923*rad[0]*np.cos(fi-np.pi/8), 0.923*rad[0]*np.sin(fi-np.pi/8)])
            vertices_plane.append([0, rad[0]*np.cos(fi), rad[0]*np.sin(fi)])

        #next circles
        for r in rad[1:]:
            for fi in angle:
                vertices_plane.append([0, r*np.cos(fi), r*np.sin(fi)])
        return vertices_plane

    planeVert = planeVertices()
    vertices = []
    for i in range(numberOfPlanes):
        for vertex in planeVert:
            vertices.append([i, vertex[1], vertex[2]])

    return np.array(vertices)

def createBrickElements(brickVertices, numberOfPlanes, numberOfPointsOnCircle, numberOfCircles):
    def createQuadrangles(planeVertices):
        indices = []
        #cetral elements
        for i in range(1, 9):
            last_ind = 2 * i + 1
            if last_ind > 16:
                last_ind = 1
            indices.append([0, 2*i - 1, 2*i, last_ind])

        #rest of elements
        # for i in range(1, np.shape(planeVertices)[0] - numberOfPointsOnCircle + 1):
        for p in range(1, numberOfCircles):
            max_ind = (p + 1)*numberOfPointsOnCircle + 1
            for i in range(1, 17):
                if p*i + 17 < max_ind:
                    indices.append([(p-1)*16 + i, (p-1)*16 + i + 16, (p-1)*16 + i + 17, (p-1)*16 + i + 1])
                else:
                    indices.append([(p-1)*16 + i, (p-1)*16 + i + 16, (p-1)*16 + i + 17 - 16, (p-1)*16 + i + 1 - 16])
        print("kwadraty", indices)
        return indices

    planeVertices = brickVertices[0: int(np.shape(brickVertices)[0]/3), :]
    quadrangleIndices = createQuadrangles(planeVertices)

    numberOfPointsOnPlane = int(np.shape(brickVertices)[0]/3)

    hexahedronIndices = []
    for i in range(numberOfPlanes - 1):
        for quad in quadrangleIndices:
            if i == 0:
                hex = [quad[0], quad[1], quad[2], quad[3], quad[0]+numberOfPointsOnPlane,
                       quad[1]+numberOfPointsOnPlane, quad[2]+numberOfPointsOnPlane, quad[3]+numberOfPointsOnPlane]
                hexahedronIndices.append(hex)
            if i == 1:
                hex = [quad[0]+numberOfPointsOnPlane, quad[1]+numberOfPointsOnPlane,
                       quad[2]+numberOfPointsOnPlane, quad[3]+numberOfPointsOnPlane,
                       quad[0]+2*numberOfPointsOnPlane, quad[1]+2*numberOfPointsOnPlane,
                       quad[2]+2*numberOfPointsOnPlane, quad[3]+2*numberOfPointsOnPlane]
                hexahedronIndices.append(hex)
    print("Liczba węzłów: ", np.shape(brickVertices)[0])
    print("Liczba elementów skończonych: ", np.shape(hexahedronIndices)[0])
    return np.array(hexahedronIndices)

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


def drawTetragons(vertices, indices):
        nodesOnPlane = int(len(vertices[:, 0])/2)
        halfOfFiniteElements = int(len(indices[:, 0])/2)

        for hexahedronIndices in indices:
            hexahedronIndicesInt = [int(i) for i in hexahedronIndices]
            plt.plot([vertices[hexahedronIndicesInt[0], 1], vertices[hexahedronIndicesInt[1], 1]],
                     [vertices[hexahedronIndicesInt[0], 2], vertices[hexahedronIndicesInt[1], 2]])
            plt.plot([vertices[hexahedronIndicesInt[1], 1], vertices[hexahedronIndicesInt[2], 1]],
                     [vertices[hexahedronIndicesInt[1], 2], vertices[hexahedronIndicesInt[2], 2]])
            plt.plot([vertices[hexahedronIndicesInt[2], 1], vertices[hexahedronIndicesInt[3], 1]],
                     [vertices[hexahedronIndicesInt[2], 2], vertices[hexahedronIndicesInt[3], 2]])
            plt.plot([vertices[hexahedronIndicesInt[3], 1], vertices[hexahedronIndicesInt[0], 1]],
                     [vertices[hexahedronIndicesInt[3], 2], vertices[hexahedronIndicesInt[0], 2]])
            plt.xlim([-11, 11])
            plt.ylim([-11, 11])
        plt.show()

def drawHexahedrons(vertices, indices):
        nodesOnPlane = int(len(vertices[:, 0])/2)
        halfOfFiniteElements = int(len(indices[:, 0])/2)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for ind, hexahedronIndices in enumerate(indices):
            hexahedronIndicesInt = [int(i) for i in hexahedronIndices]
            #[punkt], wspolrzedna
            #pierwszy czworokat
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot([vertices[hexahedronIndicesInt[0], 0], vertices[hexahedronIndicesInt[1], 0]],
                     [vertices[hexahedronIndicesInt[0], 1], vertices[hexahedronIndicesInt[1], 1]],
                     [vertices[hexahedronIndicesInt[0], 2], vertices[hexahedronIndicesInt[1], 2]])
            ax.plot([vertices[hexahedronIndicesInt[1], 0], vertices[hexahedronIndicesInt[2], 0]],
                     [vertices[hexahedronIndicesInt[1], 1], vertices[hexahedronIndicesInt[2], 1]],
                     [vertices[hexahedronIndicesInt[1], 2], vertices[hexahedronIndicesInt[2], 2]])
            ax.plot([vertices[hexahedronIndicesInt[2], 0], vertices[hexahedronIndicesInt[3], 0]],
                     [vertices[hexahedronIndicesInt[2], 1], vertices[hexahedronIndicesInt[3], 1]],
                     [vertices[hexahedronIndicesInt[2], 2], vertices[hexahedronIndicesInt[3], 2]])
            ax.plot([vertices[hexahedronIndicesInt[3], 0], vertices[hexahedronIndicesInt[0], 0]],
                     [vertices[hexahedronIndicesInt[3], 1], vertices[hexahedronIndicesInt[0], 1]],
                     [vertices[hexahedronIndicesInt[3], 2], vertices[hexahedronIndicesInt[0], 2]])

            #drugi czworokat
            ax.plot([vertices[hexahedronIndicesInt[4], 0], vertices[hexahedronIndicesInt[5], 0]],
                     [vertices[hexahedronIndicesInt[4], 1], vertices[hexahedronIndicesInt[5], 1]],
                     [vertices[hexahedronIndicesInt[4], 2], vertices[hexahedronIndicesInt[5], 2]])
            ax.plot([vertices[hexahedronIndicesInt[5], 0], vertices[hexahedronIndicesInt[6], 0]],
                     [vertices[hexahedronIndicesInt[5], 1], vertices[hexahedronIndicesInt[6], 1]],
                     [vertices[hexahedronIndicesInt[5], 2], vertices[hexahedronIndicesInt[6], 2]])
            ax.plot([vertices[hexahedronIndicesInt[6], 0], vertices[hexahedronIndicesInt[7], 0]],
                     [vertices[hexahedronIndicesInt[6], 1], vertices[hexahedronIndicesInt[7], 1]],
                     [vertices[hexahedronIndicesInt[6], 2], vertices[hexahedronIndicesInt[7], 2]])
            ax.plot([vertices[hexahedronIndicesInt[7], 0], vertices[hexahedronIndicesInt[4], 0]],
                     [vertices[hexahedronIndicesInt[7], 1], vertices[hexahedronIndicesInt[4], 1]],
                     [vertices[hexahedronIndicesInt[7], 2], vertices[hexahedronIndicesInt[4], 2]])

            #łączenie pierwszych dwóch
            ax.plot([vertices[hexahedronIndicesInt[0], 0], vertices[hexahedronIndicesInt[4], 0]],
                     [vertices[hexahedronIndicesInt[0], 1], vertices[hexahedronIndicesInt[4], 1]],
                     [vertices[hexahedronIndicesInt[0], 2], vertices[hexahedronIndicesInt[4], 2]])
            ax.plot([vertices[hexahedronIndicesInt[1], 0], vertices[hexahedronIndicesInt[5], 0]],
                     [vertices[hexahedronIndicesInt[1], 1], vertices[hexahedronIndicesInt[5], 1]],
                     [vertices[hexahedronIndicesInt[1], 2], vertices[hexahedronIndicesInt[5], 2]])
            ax.plot([vertices[hexahedronIndicesInt[2], 0], vertices[hexahedronIndicesInt[6], 0]],
                     [vertices[hexahedronIndicesInt[2], 1], vertices[hexahedronIndicesInt[6], 1]],
                     [vertices[hexahedronIndicesInt[2], 2], vertices[hexahedronIndicesInt[6], 2]])
            ax.plot([vertices[hexahedronIndicesInt[3], 0], vertices[hexahedronIndicesInt[7], 0]],
                     [vertices[hexahedronIndicesInt[3], 1], vertices[hexahedronIndicesInt[7], 1]],
                     [vertices[hexahedronIndicesInt[3], 2], vertices[hexahedronIndicesInt[7], 2]])

        plt.show()


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

#
#brickMesh(length, numberOfPlanes, radius, circles, pointsOnCircle)
# vert = brickMesh(2, 3, 10, 3, 16)
# # print(vert)
# draw_plane(vert)
# #createBrickElements(brickVertices, numberOfPlanes, numberOfPointsOnCircle, numberOfCircles)
# ind = createBrickElements(vert, 3, 16, 3)
# drawTetragons(vert, ind)
# drawHexahedrons(vert, ind)







