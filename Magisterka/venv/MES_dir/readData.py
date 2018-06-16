import numpy as np
from MES_dir import config

#Funkcje z tego modułu wykorzystywane są do zapisywania danych do plikow
#  i wczytywania danych z plików w różnych częściach programu.

def readdata(path):
    f = open(path,'r')
    temp_k0 = []
    for line in f:
        temp = []
        for i in range(int((len(line)-2)/25)):
            temp.append(float(line[i*25 : i*25+24]))
        temp_k0.append(temp)
    return np.array(temp_k0)


def read_matricies():
    config.k = readdata(config.ROOT_DIR + '/../k')
    config.m = readdata(config.ROOT_DIR + '/../m')
    config.kl = readdata(config.ROOT_DIR + '/../kl')
    config.k0 = readdata(config.ROOT_DIR + '/../k0')
    config.kr = readdata(config.ROOT_DIR + '/../kr')
    config.ml = readdata(config.ROOT_DIR + '/../ml')
    config.m0 = readdata(config.ROOT_DIR + '/../m0')
    config.mr = readdata(config.ROOT_DIR + '/../mr')


def read_complex_vector(path):
        f = open(path, 'r')
        a = []
        for line in f:
            a.append(line)
        f.close()
        complexdata = []
        for x in a:
            complexdata.append(complex(x))
        return np.array(complexdata)


def read_interpolated_fc(path):
    vector = []
    with open(path, 'r') as file:
        for line in file:
            vector.append(float(line))
    return np.array(vector)


def read_kvect(path):
    with open(path, 'r') as file:
        k = []
        for line in file:
            k.append(float(line))

    return np.array(k)


def readEigMap(path, eigval = '0+0j',delimiter = ', '):
    f = open(path, 'r')
    a = []
    eigcomplexvalue = complex(eigval)
    eigf = False
    if(eigcomplexvalue == complex(0)):
        for line in f:
            a.append(line.split(delimiter))

    else:
        for line in f:
            # print(len(line))
            if(len(line)< 60 and eigf==False):

                if(eigcomplexvalue == complex(line)):
                    eigf = True

                    continue
            elif(len(line) > 60 and eigf==True):
                temp = line
                temp = temp.replace("[", "")
                temp = temp.replace("]", "")
                a = (temp.split(delimiter))
                break
    f.close()
    complexdata = []
    for x in a:
        complexdata.append(complex(x))
    return np.array(complexdata)


def read_complex_omega(path, mode):
    vector = []
    with open(path, 'r') as file:
        for ind, line in enumerate(file):
            if ind == mode:
                for i in range(int((len(line) - 2) / 50)):
                    vector.append(complex(line[i * 50: i * 50 + 49]))
    return np.array(vector)


def read_string_omega(path, mode):
    vector = []
    with open(path, 'r') as file:
        for ind, line in enumerate(file):
            if ind == mode:
                for i in range(int((len(line) - 2) / 50)):
                    vector.append(line[i * 50: i * 50 + 49])
    return vector


def read_test(path):
    with open(path, 'r') as file:
        for line in file:
            print(line[0:-2])


def write_matrix_to_file(filename, matrix, length = 25):
    string = "0!r:{}".format(length)
    stringn = ("{","}")
    string2 = string.join(stringn)

    # m = small_numbers_equal_zero(matrix)
    with open(filename, "w") as file:
        for row in matrix:
            for element in row:
                # to_save = str(element) + " "
                # to_save.format(width=15)
                file.write(string2.format(element))

            file.write(" \n")


def write_vector_to_file(filename, vector):
    # m = small_numbers_equal_zero(matrix)
    with open(filename, "w") as file:
        for element in vector:
                # to_save = str(element) + " "
                # to_save.format(width=15)
                # file.write("{0!s:35}".format(element))
                file.write("{}".format(element))
                file.write(" \n")



# test = readEigMap('../eig/eig_0.1060287521046555','(8.24804760615e+13-1.64828439995e+12j)')

# read_test("../eig/kvect")