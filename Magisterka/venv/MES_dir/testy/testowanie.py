import numpy as np


def transpozycja(matrix):
    return np.transpose(matrix)

if __name__ == "__main__":
    a = np.array([1, 2 - 2j, 3, 4 + 1j])

    b = transpozycja(a)
    c = []
    for elem in b:
        c.append(np.conjugate(elem))
    print(c)

