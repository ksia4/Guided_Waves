from MES_dir import config
from MES_dir.hexahedralElements import assembling8, mesh8
from MES_dir.tetrahedralElements import assembling4, mesh4
from MARC import functions
import time

# Obliczenia dla elementów 4-węzłowych
def mes4(radius, numOfCircles, numOfPointsAtFirstCircle):

    vertices = mesh4.circleMeshFull(radius, numOfCircles, numOfPointsAtFirstCircle)
    if config.show_plane:
        mesh4.drawPlane(vertices)
    if config.show_bar:
        mesh4.drawBar(vertices)


    indices = mesh4.triangulation(vertices)
    # mesh.draw_triangulation(vertices, indices)
    if config.show_elements:
        mesh4.drawTriangulation(vertices, indices)

    start = time.clock()
    config.k = assembling4.assembleGlobalStiff_matrix(vertices, indices, config.young_mod, config.poisson_coef)
    # assembling.draw_matrix_sparsity(config.k)
    print("Macierz sztywnosci gotowa")

    print("Wykonywanie: ", time.clock() - start)
    config.m = assembling4.assembleGlobalMassMatrix(vertices, indices, config.density)
    config.m_focused_rows = assembling4.focuseMatrixRows(config.m)
    # assembling.draw_matrix_sparsity(config.m)
    print("Macierz mas gotowa")
    print("wykonywanie: ", time.clock() - start)

#Obliczenia dla elementów sześciennych.
#W obecnym układzie korzysta z siatki brickMesh i funkcji createBrickElements.
def mes8(numberOfPlanes, radius, numberOfCircles, numberOfPointsOnCircle, addNodes):

    # brickMesh(radius, numberOfPlanes, numberOfCircles, numberOfPointsOnCircle)
    vertices = mesh8.brickMesh(radius, numberOfPlanes, numberOfCircles, numberOfPointsOnCircle)
    # createBrickElements(brickVertices, numberOfPlanes, numberOfCircles, numberOfPointsOnCircle)
    indices = mesh8.createBrickElements(vertices, 3, 3, 16)
    start = time.clock()
    config.k = assembling8.assembleGlobalStiffMatrix(vertices, indices)
    print("Macierz sztywnosci gotowa")
    print("Wykonywanie: ", time.clock() - start, " [s]")
    print("Wykonywanie: ", (time.clock() - start)/3600, " [h]")

    start = time.clock()
    config.m = assembling8.assembleGlobalMassMatrix(vertices, indices, config.density)
    config.m_focused_rows = assembling8.focuse_matrix_rows(config.m)
    print("Macierz mas gotowa")
    print("wykonywanie: ", time.clock() - start, " [s]")
    print("wykonywanie: ", (time.clock() - start)/3600, " [h]")

