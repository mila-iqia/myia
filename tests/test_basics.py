import pytest

from myia import basics
from myia.utils import ModuleNamespace


def test_iter_range():
    r = range(1, 10)
    assert basics.myia_iter(r) is r

    assert basics.myia_hasnext(r)
    assert basics.myia_next(r) == (1, range(2, 10))

    assert not basics.myia_hasnext(range(1, 1))


def test_iter_tuple():
    t = (1, 2, 3)
    assert basics.myia_iter(t) is t
    assert basics.myia_hasnext(t)
    assert basics.myia_next(t) == (1, (2, 3))

    assert not basics.myia_hasnext(())


def test_global_universe():
    h = basics.make_handle(int)
    basics.global_universe_setitem(h, 3)
    assert basics.global_universe_getitem(h) == 3
    basics.global_universe_setitem(h, 19)
    assert basics.global_universe_getitem(h) == 19


def test_apply():
    def f(x=0, y=0, z=0):
        return (x, y, z)

    assert basics.apply(f, (), ()) == (0, 0, 0)
    assert basics.apply(f, ((),), ()) == (0, 0, 0)
    assert basics.apply(f, ((18,),), ()) == (18, 0, 0)
    assert basics.apply(f, ((1, 2),), ()) == (1, 2, 0)
    assert basics.apply(f, ((1, 2), (3,)), ()) == (1, 2, 3)
    assert basics.apply(f, ((1,),), ({"y": 2}, {"z": 3})) == (1, 2, 3)


def test_partial():
    def add(x, y):
        return x + y

    assert basics.partial(add, 4)(5) == 9


def test_make_tuple():
    assert basics.make_tuple() == ()
    assert basics.make_tuple(1, 2, 3) == (1, 2, 3)


def test_make_list():
    assert basics.make_list() == []
    assert basics.make_list(1, 2, 3) == [1, 2, 3]


def test_make_set():
    assert basics.make_set() == set()
    assert basics.make_set(1, 2, 1, 1, 3) == {1, 2, 3}


def test_make_dict():
    assert basics.make_dict() == {}
    assert basics.make_dict("a", 1, "b", 2) == {"a": 1, "b": 2}


def test_switch():
    assert basics.switch(True, 1, 2) == 1
    assert basics.switch(False, 1, 2) == 2


def test_user_switch():
    assert basics.user_switch(True, 1, 2) == 1
    assert basics.user_switch(False, 1, 2) == 2


def test_raise_():
    with pytest.raises(TypeError):
        basics.raise_(TypeError("test"))


def test_return_():
    assert basics.return_(1234) == 1234


hello = "world"


def test_resolve():
    ns = ModuleNamespace("tests.test_basics")
    assert basics.resolve(ns, "hello") == "world"
