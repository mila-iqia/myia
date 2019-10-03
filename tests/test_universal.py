
from myia import myia
from myia.lib import core, HandleInstance, Empty
from myia.operations import handle, universe_getitem, universe_setitem, \
    handle_get, handle_set


def add_one(x):
    # Not universal
    return x + 1


@core(use_universe=True)
def increment(h):
    return handle_set(h, add_one(handle_get(h)))


def test_increment():

    @myia(use_universe=True)
    def plus4(x):
        h = handle(x)
        increment(h)
        increment(h)
        increment(h)
        increment(h)
        return handle_get(h)

    assert plus4(3) == 7
    assert plus4(10) == 14


def test_increment_loop():

    @myia(use_universe=True)
    def plus(x, y):
        h = handle(x)
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return handle_get(h)

    assert plus(3, 4) == 7
    assert plus(10, 13) == 23


def test_increment_recursion():

    @myia(use_universe=True)
    def length(h, xs):
        if not isinstance(xs, Empty):
            increment(h)
            length(h, xs.tail)
        return handle_get(h)

    h = HandleInstance(0)
    assert length(h, [1, 2, 3, 4]) == 4


def test_give_handle():

    @myia(use_universe=True)
    def plus(h, y):
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return handle_get(h)

    h1 = HandleInstance(0)
    h2 = HandleInstance(0)

    # handle is updated automatically
    assert plus(h1, 4) == 4
    assert plus(h2, 9) == 9
    assert plus(h1, 30) == 34
