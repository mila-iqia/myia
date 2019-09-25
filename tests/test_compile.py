
from myia.abstract import from_value
from myia.operations import (
    array_getitem,
    array_setitem,
    bool_and,
    partial,
    primitives as P,
    scalar_add,
    tagged,
)
from myia.pipeline import standard_pipeline

from .common import MA, MB, Point, make_tree, sumtree
from .multitest import mt, run

run_no_opt = run.configure(
    pipeline=standard_pipeline.configure({'opt.phases.main': []})
)


@run(2, 3)
def test_simple(x, y):
    return x + y


@run(42)
def test_constant(x):
    return x == 42


@mt(
    run(False, True),
    run(True, True),
    run(True, False),
    run(False, False)
)
def test_bool_and(x, y):
    return bool_and(x, y)


@mt(
    run(22),
    run(3.0),
)
def test_dict(v):
    return {'x': v}


@run({'x': 22, 'y': 3.0})
def test_dict_getitem(d):
    return d['x']


@mt(
    run(33, 42),
    run(42, 33),
)
def test_if(x, y):
    if x > y:
        return x - y
    else:
        return y - x


@mt(
    run(33, 42),
    run(44, 42),
)
def test_if_nottail(x, y):
    def cap(x):
        if x > 42:
            x = 42
        return x
    return y - cap(x)


@run(42, 33)
def test_call(x, y):
    def f(x):
        return x * x

    return f(x) + f(y)


@run(42)
def test_tailcall(x):
    def fsum(x, a):
        if x == 1:
            return a
        else:
            return fsum(x - 1, a + x)
    return fsum(x, 1)


@mt(
    run(-1),
    run(1),
)
def test_callp(x):
    def fn(f, x):
        return f(x)

    def f(x):
        return -x

    return fn(f, -42)


@run(True, 42, 33)
def test_call_hof(c, x, y):
    def f1(x, y):
        return x + y

    def f2(x, y):
        return x * y

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, y) + choose(not c)(x, y)


@run_no_opt(15, 17)
def test_partial_prim(x, y):
    return partial(scalar_add, x)(y)


def test_switch_nontail():
    def fn(x, y):
        def f1():
            return x

        def f2():
            return y

        a = P.switch(x > y, f1, f2)()
        return a * a

    i64 = from_value(1, broaden=True)
    argspec = (i64, i64)
    myia_fn = standard_pipeline.run(input=fn,
                                    argspec=argspec)['output']

    for test in [(6, 23, 23**2), (67, 23, 67**2)]:
        *args, expected = test
        assert myia_fn(*args) == expected


@mt(
    run(None),
    run(42),
)
def test_is_(x):
    return x is None


@mt(
    run(None),
    run(42),
)
def test_is_not(x):
    return x is not None


@run(make_tree(3, 1))
def test_sumtree(t):
    return sumtree(t)


@mt(
    run(1, 1.7, Point(3, 4), (8, 9)),
    run(0, 1.7, Point(3, 4), (8, 9)),
    run(-1, 1.7, Point(3, 4), (8, 9)),
)
def test_tagged(c, x, y, z):
    if c > 0:
        return tagged(x)
    elif c == 0:
        return tagged(y)
    else:
        return tagged(z)


@mt(
    run('hey', 2),
    run('idk', 5),
)
def test_string_eq(s, x):
    if s == 'idk':
        x = x + 1
    return x


@mt(
    run('hey', 2),
    run('idk', 5),
)
def test_string_ne(s, x):
    if s != 'idk':
        x = x + 1
    return x


@run('hey')
def test_string_return(s):
    return s


@run(MA(4, 5))
def test_array_getitem(x):
    return array_getitem(x, (0, 1), (3, 5), (2, 3))


@run(MA(4, 5), MB(2, 2))
def test_array_setitem(x, v):
    return array_setitem(x, (0, 1), (3, 5), (2, 3), v)
