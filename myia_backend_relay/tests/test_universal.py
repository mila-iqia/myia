from dataclasses import dataclass

import pytest

from myia import myia
from myia.compile.backends import load_backend
from myia.lib import Empty, HandleInstance, core
from myia.operations import cell_get, cell_set, make_cell
from myia.pipeline import standard_pipeline, steps

load_backend("relay")


upipeline = standard_pipeline.insert_after("parse", resolve=steps.step_resolve)

umyia = myia(
    use_universe=True,
    backend="relay",
    backend_options={"exec_kind": "debug"},
    pipeline=upipeline,
)


def add_one(x):
    # Not universal, but should work from universal function
    return x + 1


@core(use_universe=True)
def increment(h):
    return cell_set(h, add_one(cell_get(h)))


def test_increment():
    @umyia
    def plus4(x):
        h = make_cell(x)
        increment(h)
        increment(h)
        increment(h)
        increment(h)
        return cell_get(h)

    assert plus4(3) == 7
    assert plus4(10) == 14


def test_increment_interleave():
    @umyia
    def plus2(x, y):
        h1 = make_cell(x)
        h2 = make_cell(y)
        increment(h1)
        increment(h2)
        increment(h1)
        increment(h2)
        return cell_get(h1), cell_get(h2)

    assert plus2(3, 6) == (5, 8)
    assert plus2(10, -21) == (12, -19)


def test_increment_loop():
    @umyia
    def plus(x, y):
        h = make_cell(x)
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return cell_get(h)

    assert plus(3, 4) == 7
    assert plus(10, 13) == 23


def test_increment_recursion():
    @umyia
    def length(h, xs):
        if not isinstance(xs, Empty):
            increment(h)
            length(h, xs.tail)
        return cell_get(h)

    h = HandleInstance(0)
    hb = length.to_device(h)
    assert length(hb, [1, 2, 3, 4]) == 4


def test_give_handle():
    @umyia
    def plus(h, y):
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return cell_get(h)

    h1 = HandleInstance(0)
    h2 = HandleInstance(0)

    hb1 = plus.to_device(h1)
    hb2 = plus.to_device(h2)

    # handle is updated automatically
    assert plus(hb1, 4) == 4
    assert plus(hb2, 9) == 9
    assert plus(hb1, 30) == 34


@pytest.mark.xfail(reason="Backend does not properly update free handles")
def test_handle_free_variable():
    h = HandleInstance(0)

    @umyia
    def plus(y):
        i = y
        while i > 0:
            increment(h)
            i = i - 1
        return cell_get(h)

    # handle is updated automatically
    assert plus(4) == 4
    assert plus(30) == 34


def test_return_handle():
    @umyia
    def plus2(h):
        increment(h)
        increment(h)
        return h

    h = HandleInstance(0)
    hb = plus2.to_device(h)
    # This might return a BackendValue later but it seems
    # to return the handle for now.
    h2 = plus2(hb)
    assert h2.state == 2


@dataclass
class Counter:
    _count: object

    def __init__(self):
        self._count = HandleInstance(0)

    @core(use_universe=True)
    def increment(self, inc):
        curr = cell_get(self._count)
        rval = curr + inc
        cell_set(self._count, rval)
        return rval

    @core(use_universe=True)
    def value(self):
        return cell_get(self._count)


def test_count():
    @umyia
    def calc(counter, n):
        for i in range(n):
            counter.increment(i + 1)
        return counter.value()

    cnt = calc.to_device(Counter())
    assert calc(cnt, 5) == 15


@pytest.mark.xfail(reason="Backend does not find handles in dataclasses")
def test_count_keepstate():
    @umyia
    def calc(counter, n):
        for i in range(n):
            counter.increment(i + 1)
        return counter.value()

    cnt = calc.to_device(Counter())
    assert calc(cnt, 5) == 15
    assert calc(cnt, 5) == 30
