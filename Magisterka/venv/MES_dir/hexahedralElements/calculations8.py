import sympy as sp
import numpy as np
import numpy.linalg as la
import scipy.integrate as integr
from MES_dir import config

# UWAGA! Wszystko dla 8-wierzchołkowych elementów 3D

#Oblicza lokalna macierz sztywności
def localStiffMatrix(elementVertices):
    ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

    def naturalShapeFunctions():

        N = [(1 / 8) * (1 - ksi) * (1 - eta) * (1 - dzeta),
             (1 / 8) * (1 + ksi) * (1 - eta) * (1 - dzeta),
             (1 / 8) * (1 + ksi) * (1 + eta) * (1 - dzeta),
             (1 / 8) * (1 - ksi) * (1 + eta) * (1 - dzeta),
             (1 / 8) * (1 - ksi) * (1 - eta) * (1 + dzeta),
             (1 / 8) * (1 + ksi) * (1 - eta) * (1 + dzeta),
             (1 / 8) * (1 + ksi) * (1 + eta) * (1 + dzeta),
             (1 / 8) * (1 - ksi) * (1 + eta) * (1 + dzeta)]

        return N

    def naturalBMatrix():

        N = naturalShapeFunctions()
        bRows = 6
        B = []
        for i in range(bRows):
            temp = []
            for n in N:
                if i == 0:
                    temp.append(sp.diff(n, ksi))
                    temp.append(0)
                    temp.append(0)
                if i == 1:
                    temp.append(0)
                    temp.append(sp.diff(n, eta))
                    temp.append(0)
                if i == 2:
                    temp.append(0)
                    temp.append(0)
                    temp.append(sp.diff(n, dzeta))
                if i == 3:
                    temp.append(sp.diff(n, eta))
                    temp.append(sp.diff(n, ksi))
                    temp.append(0)
                if i == 4:
                    temp.append(sp.diff(n, dzeta))
                    temp.append(0)
                    temp.append(sp.diff(n, ksi))
                if i == 5:
                    temp.append(0)
                    temp.append(sp.diff(n, dzeta))
                    temp.append(sp.diff(n, eta))
            B.append(temp)
        return np.array(B)

    def dMatrix(youngModulus, poissonCoefficient):
        a = youngModulus / ((1 + poissonCoefficient) * (1 - 2 * poissonCoefficient))
        b1 = (1 - poissonCoefficient) * a
        b2 = poissonCoefficient * a
        b3 = ((1 - 2 * poissonCoefficient) / 2) * a
        matrix = [[b1, b2, b2, 0, 0, 0], [b2, b1, b2, 0, 0, 0], [b2, b2, b1, 0, 0, 0],
                  [0, 0, 0, b3, 0, 0], [0, 0, 0, 0, b3, 0], [0, 0, 0, 0, 0, b3]]
        return np.array(matrix)

    def jacobian(elementVertices):

        def coordinateChangeModel():

            temp = 0
            for vertex, shapeFc in zip(elementVertices[:, 0], naturalShapeFunctions()):
                temp += vertex * shapeFc
            xInNaturalCoor = temp
            temp = 0

            for vertex, shapeFc in zip(elementVertices[:, 1], naturalShapeFunctions()):
                temp += vertex * shapeFc
            yInNaturalCoor = temp
            temp = 0

            for vertex, shapeFc in zip(elementVertices[:, 2], naturalShapeFunctions()):
                temp += vertex * shapeFc
            zInNaturalCoor = temp

            return xInNaturalCoor, yInNaturalCoor, zInNaturalCoor

        xInNatural, yInNatural, zInNatural = coordinateChangeModel()

        jacobianMatrix = [[], [], []]
        jacobianMatrix[0].append(sp.diff(xInNatural, ksi))
        jacobianMatrix[0].append(sp.diff(yInNatural, ksi))
        jacobianMatrix[0].append(sp.diff(zInNatural, ksi))

        jacobianMatrix[1].append(sp.diff(xInNatural, eta))
        jacobianMatrix[1].append(sp.diff(yInNatural, eta))
        jacobianMatrix[1].append(sp.diff(zInNatural, eta))

        jacobianMatrix[2].append(sp.diff(xInNatural, dzeta))
        jacobianMatrix[2].append(sp.diff(yInNatural, dzeta))
        jacobianMatrix[2].append(sp.diff(zInNatural, dzeta))

        jacobianMatrix = sp.Matrix(jacobianMatrix)
        return sp.det(jacobianMatrix)  # jacobian

    bMatrix = naturalBMatrix()
    dMatrix = dMatrix(config.young_mod, config.poisson_coef)

    bTransposed = bMatrix.transpose()
    temp = bTransposed.dot(dMatrix)

    matrixToIntegrate = temp.dot(bMatrix) * jacobian(elementVertices) / 1e6

    yBot = lambda ksi: -1
    yTop = lambda ksi: 1
    zBot = lambda ksi, eta: -1
    zTop = lambda ksi, eta: 1

    # Integration
    matrixAfterIntegration = []

    for row in matrixToIntegrate:
        integr_row = []

        for elem in row:
            if elem != 0:
                lam_elem = sp.lambdify((ksi, eta, dzeta), elem)

                temp = integr.tplquad(lam_elem, -1, 1, yBot, yTop, zBot, zTop)[0]

            if elem == 0:
                temp = 0
            integr_row.append(temp)
        matrixAfterIntegration.append(integr_row)

    return np.array(matrixAfterIntegration) * 1e6 #stiff matrix

#Oblicza lokalna macierz mas
def localMassMatrix(elementVertices):
        ksi, eta, dzeta = sp.symbols('ksi, eta, dzeta')

        def naturalShapeFunctions():

            N = [(1 / 8) * (1 - ksi) * (1 - eta) * (1 - dzeta),
                 (1 / 8) * (1 + ksi) * (1 - eta) * (1 - dzeta),
                 (1 / 8) * (1 + ksi) * (1 + eta) * (1 - dzeta),
                 (1 / 8) * (1 - ksi) * (1 + eta) * (1 - dzeta),
                 (1 / 8) * (1 - ksi) * (1 - eta) * (1 + dzeta),
                 (1 / 8) * (1 + ksi) * (1 - eta) * (1 + dzeta),
                 (1 / 8) * (1 + ksi) * (1 + eta) * (1 + dzeta),
                 (1 / 8) * (1 - ksi) * (1 + eta) * (1 + dzeta)]

            return N

        def jacobian(elementVertices):

            def coordinateChangeModel():

                temp = 0
                for vertex, shapeFc in zip(elementVertices[:, 0], naturalShapeFunctions()):
                    temp += vertex * shapeFc
                xInNaturalCoor = temp
                temp = 0

                for vertex, shapeFc in zip(elementVertices[:, 1], naturalShapeFunctions()):
                    temp += vertex * shapeFc
                yInNaturalCoor = temp
                temp = 0

                for vertex, shapeFc in zip(elementVertices[:, 2], naturalShapeFunctions()):
                    temp += vertex * shapeFc
                zInNaturalCoor = temp

                return xInNaturalCoor, yInNaturalCoor, zInNaturalCoor

            xInNatural, yInNatural, zInNatural = coordinateChangeModel()

            jacobianMatrix = [[], [], []]
            jacobianMatrix[0].append(sp.diff(xInNatural, ksi))
            jacobianMatrix[0].append(sp.diff(yInNatural, ksi))
            jacobianMatrix[0].append(sp.diff(zInNatural, ksi))

            jacobianMatrix[1].append(sp.diff(xInNatural, eta))
            jacobianMatrix[1].append(sp.diff(yInNatural, eta))
            jacobianMatrix[1].append(sp.diff(zInNatural, eta))

            jacobianMatrix[2].append(sp.diff(xInNatural, dzeta))
            jacobianMatrix[2].append(sp.diff(yInNatural, dzeta))
            jacobianMatrix[2].append(sp.diff(zInNatural, dzeta))

            jacobianMatrix = sp.Matrix(jacobianMatrix)

            return sp.det(jacobianMatrix)  # jacobian

        def matrixWithShapeFunctions():
            shapeFc = naturalShapeFunctions()
            matrix = []
            for i in range(3): #będą zawsze 3 wiersze
                temp = []
                for n in shapeFc:
                    if i == 0:
                        temp.append(n)
                        temp.append(0)
                        temp.append(0)
                    if i == 1:
                        temp.append(0)
                        temp.append(n)
                        temp.append(0)
                    if i == 2:
                        temp.append(0)
                        temp.append(0)
                        temp.append(n)

                matrix.append(temp)
            return np.array(matrix)

        nMatrix = matrixWithShapeFunctions()

        nTransposed = nMatrix.transpose()

        matrixToIntegrate = nTransposed.dot(nMatrix) * jacobian(elementVertices) * config.density

        yBot = lambda ksi: -1
        yTop = lambda ksi: 1
        zBot = lambda ksi, eta: -1
        zTop = lambda ksi, eta: 1

        # Integration
        matrixAfterIntegration = []
        for row in matrixToIntegrate:
            integr_row = []
            for elem in row:
                if elem != 0:
                    lam_elem = sp.lambdify((ksi, eta, dzeta), elem)

                    temp = integr.tplquad(lam_elem, -1, 1, yBot, yTop, zBot, zTop)[0]

                if elem == 0:
                    temp = 0

                integr_row.append(temp)
            matrixAfterIntegration.append(integr_row)
        return np.array(matrixAfterIntegration) #mass matrix



