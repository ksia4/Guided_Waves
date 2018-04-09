import numpy as np
from MES_dir import readData as rd


def normalization(vect):
    maxlen = 0
    normvect = []
    for val in vect:
        vectlen = np.sqrt(np.real(val)**2 + np.imag(val)**2)
        if(vectlen > maxlen):
            maxlen = vectlen

    for val in vect:
        normvect.append(complex(val/maxlen))
    return normvect


def createNormFile(kvectPath):
    with open(kvectPath, 'r') as file: #otwieramy plik z zapisanymi k
        for line in file: #Dla każdego k otwieramy odpowiedni plik
            vectnromeigvect = []
            eigpath = "../eig/eig_{}".format(line[0:-2])
            normeigpath = "../eig/normeig/normeig_{}".format(line[0:-2])
            with open(eigpath, 'r') as eigfile:
                for ind, eigline in enumerate(eigfile):
                    if(ind%2==0): #jeśli to jest parzysty wiersz => to jest wartość własna
                        eigval = eigline[0:-2]
                        normeigvect = normalization(rd.readEigMap(eigpath,eigval))
                        vectnromeigvect.append(complex(eigval))
                        vectnromeigvect.append(normeigvect)
            rd.write_vector_to_file(normeigpath,vectnromeigvect)


