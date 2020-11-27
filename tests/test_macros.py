import pytest

from myia import myia
from myia.abstract import macro, myia_static
from myia.testing.multitest import bt
from myia.utils import InferenceError, InternalInferenceError


@macro
async def mackerel(info):
    pass


def test_repr():
    assert repr(mackerel) == "<Macro mackerel>"


@bt()
def test_bad_macro(backend):
    from myia.ir import Graph
    from myia.operations import primitives as P

    @macro
    async def bad(info, x):
        badg = Graph()
        badg.debug.name = "badg"
        p = badg.add_parameter()
        p.debug.name = "parameter"
        badg.output = badg.apply(P.scalar_add, p, p)
        # The return value of the macro can't directly refer to badg.output
        # because that node can only be accessed from the scope of a call to
        # badg. The error raised by Myia should reflect this.
        return info.graph.apply(P.transpose, badg.output)

    @myia(backend=backend)
    def salmon(x, y):
        return bad(x + y)

    with pytest.raises(InternalInferenceError):
        salmon(12, 13)


def test_bad_macro_2():
    with pytest.raises(TypeError):
        # Should be async def
        @macro
        def bigmac(info):
            return info.nodes()[0]


@bt()
def test_bad_macro_3(backend):
    @macro
    async def macncheese(info, x):
        # Should return a node
        return None

    @myia(backend=backend)
    def pasta(x, y):
        return macncheese(x + y)

    with pytest.raises(InferenceError):
        pasta(12, 13)


def test_bad_macro_4():
    with pytest.raises(TypeError):
        # Cannot define default arguments
        @macro
        async def bigmac(info, x=None):
            return info.nodes()[0]


@myia_static
def static_add(x, y):
    return x + y


@myia_static
def static_blah(x, y):
    return static_add(x, y)


def test_myia_static_in_myia_static():
    static_blah(1, 2)


@bt()
def test_myia_static(backend):
    @myia(backend=backend)
    def get_fourth(xs):
        return xs[static_add(1, y=2)]

    assert get_fourth((1, 2, 3, 4, 5)) == 4

    @myia(backend=backend)
    def get_fourth_bad(xs):
        return xs[static_add(1, x=2)]

    with pytest.raises(InferenceError):
        get_fourth_bad((1, 2, 3, 4, 5))

    @myia(backend=backend)
    def add1_bad(x):
        return static_add(1, x)

    with pytest.raises(InferenceError):
        add1_bad(5)
