from MES_dir import config
from MES_dir.hexahedralElements import assembling8, mesh8
from MES_dir.tetrahedralElements import assembling4, mesh4
from dispersion_curves import functions
import time

    # Obliczenia dla elementów 4-węzłowych
def mes4(length, radius, num_of_circles, num_of_points_at_c1):

    # d = functions.read_coordinates_from_DAT("rod_v4_job1_copy1.dat", 724, 3)
    # L, C, R = functions.planes_indecies(d)
    # vertices = d[L + C + R, 1:4]

    vertices = mesh4.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    if config.show_plane:
        mesh4.draw_plane(vertices)
    if config.show_bar:
        mesh4.draw_bar(vertices)


    indices = mesh4.triangulation(vertices)
    # mesh.draw_triangulation(vertices, indices)
    if config.show_elements:
        mesh4.draw_triangulation(vertices, indices)

    start = time.clock()
    config.k = assembling4.assemble_global_stiff_matrix(vertices, indices, config.young_mod, config.poisson_coef)
    # assembling.draw_matrix_sparsity(config.k)
    print("Macierz sztywnosci gotowa")

    print("Wykonywanie: ", time.clock() - start)
    config.m = assembling4.assemble_global_mass_matrix(vertices, indices, config.density)
    config.m_focused_rows = assembling4.focuse_matrix_rows(config.m)
    # assembling.draw_matrix_sparsity(config.m)
    print("Macierz mas gotowa")
    print("wykonywanie: ", time.clock() - start)

def mes8(length, numberOfPlanes, radius, firstCircle, addNodes, circles):

    # d = functions.read_coordinates_from_DAT(config.ROOT_DIR + "/../dispersion_curves/rod_v4_job1.dat", 724, 3)
    # L, C, R = functions.planes_indecies(d)
    # vertices = d[L + C + R, 1:4]
    # numberOfPointsOnLastCircle = 19


    # vertices, numberOfPointsOnLastCircle = mesh8.circleMeshFull(length, numberOfPlanes, radius, firstCircle, addNodes, circles)
    #
    # if config.show_plane:
    #     mesh8.draw_plane(vertices)
    # if config.show_bar:
    #     mesh8.draw_bar(vertices)
    # vertices, indices = mesh8.createFiniteElements(vertices, numberOfPointsOnLastCircle, length, numberOfPlanes)
    #
    # if config.show_elements:
    #     mesh8.drawTetragons(vertices, indices)

    # brickMesh(length, numberOfPlanes, radius, circles, pointsOnCircle)
    vertices = mesh8.brickMesh(2, 3, 10, 10, 16)
    # createBrickElements(brickVertices, numberOfPlanes, numberOfPointsOnCircle, numberOfCircles)
    indices = mesh8.createBrickElements(vertices, 3, 16, 10)
    print(indices)
    start = time.clock()
    config.k = assembling8.assemble_global_stiff_matrix(vertices, indices)
    # assembling.draw_matrix_sparsity(config.k)
    print("Macierz sztywnosci gotowa")
    print("Wykonywanie: ", time.clock() - start, " [s]")
    print("Wykonywanie: ", (time.clock() - start)/3600, " [h]")

    start = time.clock()
    config.m = assembling8.assemble_global_mass_matrix(vertices, indices, config.density)
    # config.m_focused_rows = assembling8.focuse_matrix_rows(config.m)
    # assembling.draw_matrix_sparsity(config.m)
    print("Macierz mas gotowa")
    print("wykonywanie: ", time.clock() - start, " [s]")
    print("wykonywanie: ", (time.clock() - start)/3600, " [h]")

