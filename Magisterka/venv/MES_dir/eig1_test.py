import numpy as np
import numpy.linalg as la


a = np.array([[4, 1, 0],
              [0, 2, 1],
             [0, 0, -1]])
vec = np.array([1, 1, 1])
vec1 = vec/la.norm(vec)

for k in range(100):
    new_vec = np.dot(a, vec1)
    vec1 = new_vec/la.norm(new_vec)
    print(new_vec)
    print(la.norm(new_vec))
    print(np.dot(a, vec1)/vec1)
    print("")

print(la.eig(a))
