from MES_dir import config
from MES_dir import mesh, assembling
import time


def mes(length, radius, num_of_circles, num_of_points_at_c1):

    # d = functions.read_coordinates_from_DAT("rod_v4_job1_copy1.dat", 724, 3)
    # L, C, R = functions.planes_indecies(d)
    # vertices = d[L + C + R, 1:4]

    vertices = mesh.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    if config.show_plane:
        mesh.draw_plane(vertices)
    if config.show_bar:
        mesh.draw_bar(vertices)


    indices = mesh.triangulation(vertices)
    # mesh.draw_triangulation(vertices, indices)
    if config.show_elements:
        mesh.draw_triangulation(vertices, indices)

    start = time.clock()
    config.k = assembling.assemble_global_stiff_matrix(vertices, indices, config.young_mod, config.poisson_coef)
    # assembling.draw_matrix_sparsity(config.k)
    print("Macierz sztywnosci gotowa")

    print("Wykonywanie: ", time.clock() - start)
    config.m = assembling.assemble_global_mass_matrix(vertices, indices, config.density)
    config.m_focused_rows = assembling.focuse_matrix_rows(config.m)
    # assembling.draw_matrix_sparsity(config.m)
    print("Macierz mas gotowa")
    print("wykonywanie: ", time.clock() - start)

