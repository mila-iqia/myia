import os

from myia import myia


def test_pdb():
    @myia(backend="python", backend_options={"debug": True, "pdb": True})
    def f(a, b, c, d):
        x = a ** b
        y = x * c
        z = x / d
        return y + z + x

    wd = os.getcwd()
    # Change working directory to use local .pdbrc
    os.chdir(os.path.dirname(__file__))
    assert f(1.0, 2.0, 3.0, 4.0) == 4.25
    # Back to previous working directory
    os.chdir(wd)
