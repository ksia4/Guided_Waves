from MES_dir import calculations, config
from dispersion_curves import functions

vert = functions.read_coordinates_from_DAT('model8_job1.dat', 4, 3)
v = vert[0:5, 1:4]
# vert = np.array([[3, 1, 0], [-1, 2, 0], [5, -9, 0], [-1, -1, 1]])
ind = [0, 1, 2, 3]
sf = calculations.shape_functions(v, ind)
d = calculations.d_matrix_fc(config.young_mod, config.poisson_coef)
stiff = calculations.stiff_local_matrix(sf, v, ind, config.young_mod, config.poisson_coef)
print(stiff)
mass = calculations.mass_local_matrix(config.density)
print(mass)
# print("koniec")

k = functions.read_MARC_matrix('model8_job1_glstif_0001.bin', 4, 3)
print("koniec")
print("aaa")