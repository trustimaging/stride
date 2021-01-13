
import time
import numpy as np

# from mosaic2 import init, stop
# from mosaic2 import tessera


class Solver1:
    def __init__(self, data):
        self.data = data

    def solve(self, data):
        print('Solve 1')
        self.data = self.data + data

        time.sleep(10)

        return self.data

    def solve_more(self):
        print('Solve More 1')
        time.sleep(5)


class Solver2:
    def __init__(self):
        self.data = 0

    def solve(self, data):
        print('Solve 2')
        self.data = data*2

        time.sleep(10)

        return self.data

    def solve_more(self):
        print('Solve More 2')
        time.sleep(5)


if __name__ == '__main__':
    array = np.zeros((1024, 1024, 1024), dtype=np.float32, order='C')

    # Create objects
    solver_1 = Solver1(array)
    solver_2 = Solver2()

    # Case 1
    result_1 = solver_1.solve(array)
    result_2 = solver_2.solve(array)

    # Case 2
    result_1 = solver_1.solve(array)
    result_2 = solver_2.solve(result_1)

    # Case 3
    solver_1.solve_more()
    solver_2.solve_more()
