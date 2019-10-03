
from myia import myia
from myia.lib import core, HandleInstance
from myia.operations import handle, universe_getitem, universe_setitem


@core(universal=True)
def increment(U, h):
    v = universe_getitem(U, h)
    U = universe_setitem(U, h, v + 1)
    return U, None


@core(universal=True)
def get(U, h):
    return U, universe_getitem(U, h)


def test_increment():

    @myia(universal=True)
    def plus4(x):
        h = handle(x)
        increment(h)
        increment(h)
        increment(h)
        increment(h)
        return get(h)

    assert plus4(3) == 7
    assert plus4(10) == 14


def test_increment_loop():

    @myia(universal=True)
    def plus(x, y):
        h = handle(x)
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return get(h)

    assert plus(3, 4) == 7
    assert plus(10, 13) == 23


def test_give_handle():

    @myia(universal=True)
    def plus(h, y):
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return get(h)

    h1 = HandleInstance(0)
    h2 = HandleInstance(0)

    # handle is updated automatically
    assert plus(h1, 4) == 4
    assert plus(h2, 9) == 9
    assert plus(h1, 30) == 34
