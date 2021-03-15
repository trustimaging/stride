
import time
import numpy as np

import mosaic
from mosaic import tessera


@tessera
class Solver1:
    def __init__(self, data):
        self.data = data

    def solve(self, data):
        print('Solve 1')
        self.data = self.data + data

        time.sleep(10)
        print('Done 1')

        return self.data

    def solve_more(self):
        print('Solve More 1')
        time.sleep(5)
        print('Done More 1')


@tessera
class Solver2:
    def __init__(self):
        self.data = 0

    def solve(self, data):
        print('Solve 2')
        self.data = data*2

        time.sleep(10)
        print('Done 2')

        return self.data

    def solve_more(self):
        print('Solve More 2')
        time.sleep(5)
        print('Done More 2')


async def main(runtime):
    array = np.zeros((1024, 1024, 1), dtype=np.float32)

    # These objects will be created remotely
    solver_1 = await Solver1.remote(array)
    solver_2 = await Solver2.remote()

    # These will run in parallel
    # The calls will return immediately by creating a remote
    # task
    start = time.time()
    task_1 = await solver_1.solve(array)
    task_2 = await solver_2.solve(array)

    # Do some other work

    # Wait until the remote tasks are finished
    await task_1
    await task_2

    # The results of the tasks stay in the remote worker
    # until we request it back
    result_1 = await task_1.result()
    result_2 = await task_2.result()

    print(result_1.shape)
    print(result_2.shape)
    print(time.time() - start)

    # These will wait for each other because
    # their results depend on each other
    start = time.time()
    task_1 = await solver_1.solve(array)
    task_2 = await solver_2.solve(task_1)

    # Do some other work

    # Wait until the remote tasks are finished
    # Now we only need to wait for the second task
    await task_2
    print(time.time() - start)

    # These will also wait for each other
    start = time.time()
    task_1 = await solver_1.solve_more()
    task_2 = await solver_2.solve_more(task_1.outputs.done)

    # Do some other work

    # Wait until the remote tasks are finished
    # Now we only need to wait for the second task
    await task_2
    print(time.time() - start)


if __name__ == '__main__':
    mosaic.run(main)
