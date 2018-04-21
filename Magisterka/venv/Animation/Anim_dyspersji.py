import sympy as sp
import numpy as np
from MES_dir import MES, config, dispersion, mesh
import matplotlib.pyplot as plt
import matplotlib.colors as color

BASECOLOR = (230/255, 230/255, 250/255)


def draw_bar(vertices, num_of_points_in_one_piece, length):
    print(len(vertices))
    planes = len(vertices)/num_of_points_in_one_piece
    check = int(planes)
    if planes-check != 0:
        print("Coś tu się źle policzyło...")
        exit(0)
    print(check)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=BASECOLOR)

    # ax.scatter(vertices[0:num_of_points_in_one_piece, 0], vertices[0:num_of_points_in_one_piece, 1], vertices[0:num_of_points_in_one_piece, 2], color='red')
    # ax.scatter(vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 0], vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 1], vertices[num_of_points_in_one_piece:2*num_of_points_in_one_piece, 2], color='white', edgecolor='red')

    lim = int(length/2)
    ax.set_xlim([-1, length+1])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    plt.show()

if __name__ == "__main__":
    # print("Wpisz wartość: ")
    # print("1 - rysowanie krzywy dyspersji z wykorzystaniem MES")
    # print("2 - rysowanie krzywych dyspersji z ostatnio policzonych danych")
    # text = input()

    # if text == '1':
        # parametry preta
    length = 100
    # num_of_planes = length*100
    # factor = 0.001
    radius = 10
    num_of_circles = 6
    num_of_points_at_c1 = 6

    # wektor liczby falowej
    config.kvect_min = 1e-10
    config.kvect_max = np.pi / 2
    config.kvect_no_of_points = 101

    # rysowanie wykresow
    # config.show_plane = False
    # config.show_bar = True
    # config.show_elements = False

    # obliczenia
    plane = mesh.circle_mesh_full(1, radius, num_of_circles, num_of_points_at_c1)
    vertices = mesh.circle_mesh_full(length, radius, num_of_circles, num_of_points_at_c1)
    draw_bar(vertices, len(plane), length)
