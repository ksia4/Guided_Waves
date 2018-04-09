def hex_plane_vertices(x):
    verticies = [[x, 0, 0]]
    # angle = [30, 0, 30, 15, 45, 0, 30]
    angle = [30, 0, 30, 10, 50]
    angle1 = []
    for fi in angle:
        angle1.append(fi*(2*np.pi)/360)
    # rad = [1, np.sqrt(3), 2, np.sqrt(7), np.sqrt(7), 2*np.sqrt(3), 2*np.sqrt(3)]
    rad = [1, np.sqrt(3), 2, np.sqrt(7), np.sqrt(7)]
    for r, fi in zip(rad, angle1):
        for i in range(6):
            fi = fi + np.pi/3
            y = r*np.cos(np.pi/2-fi)
            z = r*np.sin(np.pi/2-fi)
            vertex = [x, y, z]
            verticies.append(vertex)
            print(vertex)
            print(np.sqrt(y**2 + z**2))
    return np.array(verticies)


def hex_mesh(length):
    vertices = [[0, 0, 0]]
    for i in range(length):
        vertices = np.append(vertices, hex_plane_verticies(i), axis=0)
    return np.array(vertices)


def equal_size_plane_vertices(x, radius, density, number_of_edge_points):
    vertices = []
    y_interval = np.linspace(-radius, radius, density)
    z_interval = y_interval
    for y in y_interval:
        for z in z_interval:
            if y**2 + z**2 < radius**2 :
                vertices.append([x, y, z])

    angles = np.linspace(0, 2*np.pi, number_of_edge_points)

    for fi in angles:
        y = radius*np.cos(fi)
        z = radius*np.sin(fi)
        vertices.append([x, y, z])

    return np.array(vertices)


def equal_size_mesh(length, planes, radius, density, number_of_edge_points):
    length_list = np.linspace(0, length, planes)
    vertices = equal_size_plane_vertices(0, radius, density, number_of_edge_points)
    length_list1 = np.delete(length_list, 0)

    for l in length_list1:
        vertices = np.append(vertices, equal_size_plane_vertices(l, radius, density, number_of_edge_points), axis=0)

    return np.array(vertices)
