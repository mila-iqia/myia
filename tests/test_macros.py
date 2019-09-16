
import pytest

from myia import myia
from myia.abstract import macro, myia_static
from myia.utils import InferenceError, InternalInferenceError


@macro
async def mackerel(info):
    pass


def test_repr():
    assert repr(mackerel) == '<Macro mackerel>'


def test_bad_macro():
    from myia.ir import Graph
    from myia.prim import ops as P

    @macro
    async def bad(info):
        badg = Graph()
        badg.debug.name = 'badg'
        p = badg.add_parameter()
        p.debug.name = 'parameter'
        badg.output = badg.apply(P.scalar_add, p, p)
        # The return value of the macro can't directly refer to badg.output
        # because that node can only be accessed from the scope of a call to
        # badg. The error raised by Myia should reflect this.
        return info.graph.apply(P.transpose, badg.output)

    @myia
    def salmon(x, y):
        return bad(x + y)

    with pytest.raises(InternalInferenceError):
        salmon(12, 13)


def test_bad_macro_2():
    with pytest.raises(TypeError):
        @macro
        def bigmac(info):
            return info.args[0]


def test_bad_macro_3():
    @macro
    async def macncheese(info):
        return None

    @myia
    def pasta(x, y):
        return macncheese(x + y)

    with pytest.raises(InferenceError):
        pasta(12, 13)


@myia_static
def static_add(x, y):
    return x + y


@myia_static
def static_blah(x, y):
    return static_add(x, y)


def test_myia_static_in_myia_static():
    static_blah(1, 2)


def test_myia_static():
    @myia
    def get_fourth(xs):
        return xs[static_add(1, y=2)]

    assert get_fourth((1, 2, 3, 4, 5)) == 4

    @myia
    def get_fourth_bad(xs):
        return xs[static_add(1, x=2)]

    with pytest.raises(InferenceError):
        get_fourth_bad((1, 2, 3, 4, 5))

    @myia
    def add1_bad(x):
        return static_add(1, x)

    with pytest.raises(InferenceError):
        add1_bad(5)
