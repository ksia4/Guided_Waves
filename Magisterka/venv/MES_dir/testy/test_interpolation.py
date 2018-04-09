from unittest import TestCase, main
import numpy as np
from MES_dir import excitability as exc

class TestInterpolation(TestCase):


    def setUp(self):
        pass


    def test_second_order(self):
        arguments = np.array([1, 2, 3])
        values = np.array([2, 5, 7])

        polynomial = exc.interpolation(arguments, values)
        check = np.array([-2., 4.5, -0.5])

        temp = 0
        for i in range(len(arguments)):
            if check[i] - 1e-10 > polynomial[i] or polynomial[i] > check[i] + 1e-10:
                self.assertFalse(self)
                temp += 1
                break
        if temp == 0:
            self.assertTrue(self)


if __name__ == "__main__":
    main()