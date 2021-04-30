"""Test Python backend with option pdb=True.

It seems pytest-cov does not work well if PDB is imported during pytest runs.
To deal with it, code with PDB is executed in a separate process. This seems
enough to make coverage work again.
"""
import io
import operator
import multiprocessing
import os
from myia.utils.info import enable_debug
from myia.parser import parse
from myia.compile.backends.python.python import compile_graph


def parse_and_compile(function):
    with enable_debug():
        graph = parse(function)
    output = io.StringIO()
    fn = compile_graph(graph, debug=output, pdb=True)
    output = output.getvalue()
    print()
    print(output)
    return fn, output


def run_pdb(return_cell, *args):
    # Myia-compiled function result will be saved in return_cell (shared list).

    def f(a, b, c, d):
        x = a ** b
        y = operator.mul(x, c)
        z = x / d
        return y + z + x

    fn, output = parse_and_compile(f)
    assert output == """# Dynamic external import: operator

def f(a, b, c, d):
  x = a ** b
  _apply1 = operator.mul
  y = _apply1(x, c)
  z = x / d
  _apply2 = y + z
  return _apply2 + x
"""

    wd = os.getcwd()
    # Change working directory to use local .pdbrc
    os.chdir(os.path.dirname(__file__))
    return_cell.append(fn(*args))
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
