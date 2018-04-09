from unittest import TestCase, main
from MES_dir import excitability as exc
import numpy as np


class TestHermitian_transpose(TestCase):

    def setUp(self):
        pass

    def test_2x2(self):
        matrix_2x2 = np.array([[1-1j, 2+3j],
                               [3, 4]])
        final_matrix_2x2 = np.array([[1+1j, 3+0j],
                                    [2-3j, 4+0j]])
        temp = 0
        for row, frow in zip(exc.hermitian_transpose(matrix_2x2), final_matrix_2x2):
            for elem, felem in zip(row, frow):
                if elem != felem:
                    self.assertFalse(self)
                    break
                temp += 1
        if temp == 4:
            self.assertTrue(self)

    def test_vector(self):

        vector = np.array([1+2j, 2-3j, 3])
        final_vector = np.array([1-2j, 2+3j, 3])

        temp = 0

        for elem, felem in zip(exc.hermitian_transpose(vector), final_vector):
            if elem != felem:
                self.assertFalse(self)
                break
            temp +=1

        if temp == 3:
            self.assertTrue(self)


if __name__ == "__main__":
    main()