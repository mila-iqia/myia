"""Test Python backend with option pdb=True.

It seems pytest-cov does not work well if PDB is imported during pytest runs.
To deal with it, code with PDB is executed in a separate process. This seems
enough to make coverage work again.
"""
import multiprocessing
import os

from myia import myia


def run_pdb(return_cell, *args):
    # Myia-compiled function result will be saved in return_cell (shared list).

    @myia(backend="python", backend_options={"debug": True, "pdb": True})
    def f(a, b, c, d):
        x = a ** b
        y = x * c
        z = x / d
        return y + z + x

    wd = os.getcwd()
    # Change working directory to use local .pdbrc
    os.chdir(os.path.dirname(__file__))
    return_cell.append(f(*args))
    # Back to previous working directory
    os.chdir(wd)


def test_pdb():
    manager = multiprocessing.Manager()
    return_cell = manager.list()
    p = multiprocessing.Process(
        target=run_pdb, args=((return_cell, 1.0, 2.0, 3.0, 4.0))
    )
    p.start()
    p.join()
    assert return_cell[0] == 4.25
