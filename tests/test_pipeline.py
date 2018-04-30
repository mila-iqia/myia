
import pytest
from myia.pipeline import Symbol, Universe, PythonUniverse, PipelineGenerator


class PowerUniverse(Universe):
    def __init__(self, parent, power):
        super().__init__(parent)
        self.power = power

    def acquire(self, x):
        return x ** self.power


def f():
    pass


globbo = 42
gsym = Symbol('globbo', namespace=f'global:{__file__}')


def test_python_universe():
    u = PythonUniverse()
    with pytest.raises(NameError):
        u[gsym]
    u[f]
    assert u[gsym] == 42


def test_pipeline():
    pgen = PipelineGenerator(py=PythonUniverse,
                             pow=PowerUniverse,
                             pow2=PowerUniverse)
    p1 = pgen('py->pow', pow={'power': 2})
    assert p1[3] == 3**2
    p2 = pgen('py->pow->pow2', pow={'power': 2}, pow2={'power': 3})
    assert p2[3] == (3**2)**3
    assert p2.parent is p1
    p2_2 = pgen('py->pow->pow2', pow={'power': 2}, pow2={'power': 3})
    assert p2_2 is p2
    p2_3 = pgen('py->pow->pow2', pow={'power': 2}, pow2={'power': 3}, cache=False)
    assert p2_3 is not p2
    assert p2_3.parent is not p2.parent
    assert p2_3.parent.parent is not p2.parent.parent
    p3 = pgen('py->pow->pow2', pow={'power': 4}, pow2={'power': 3})
    assert p3[3] == (3**4)**3
    assert p3.parent is not p1


def test_pipeline_errors():
    pgen = PipelineGenerator(py=PythonUniverse,
                             pow=PowerUniverse,
                             pow2=PowerUniverse)
    with pytest.raises(NameError):
        pgen('py->pop')
    with pytest.raises(NameError):
        pgen('py->pow', pow2={})
