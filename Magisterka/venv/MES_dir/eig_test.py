import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.sparse.linalg as spla
import time

dim = 10
a = np.random.randint(-100, 100, [1000, 1000]) * 0.5 + np.random.randint(-100, 100, [1000, 1000]) * 0.1j
print(a)

start = time.clock()
npa, npb = nla.eig(a)
print(start - time.clock())

start = time.clock()
slaa, slab = sla.eig(a)
print(start - time.clock())

start = time.clock()
splaa, splab = spla.eigsh(a, 5, sigma=0, which='SM', tol=1E-2)
print(start - time.clock())
