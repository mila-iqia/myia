from myia.compile.backends.python.python import compile_graph
from myia.ir import manage
from myia.ir.utils import print_graph
from myia.parser import parse


def test_from_parse_with_if():
    def f(x):
        if x < 0:
            return x * x
        else:
            return x * x * x

    g = parse(f)
    manage(g)

    print(print_graph(g))

    cf = compile_graph(g, debug=True)
    assert cf(-2) == 4, cf(-2)
    assert cf(0) == 0, cf(0)
    assert cf(2) == 8, cf(2)


def test_from_parse():
    def f(x):
        return 2 * x + 1

    g = parse(f)
    manage(g)

    print(print_graph(g))

    cf = compile_graph(g, debug=True)
    assert cf(0) == 1, cf(0)
    assert cf(-2) == -3, cf(0)
