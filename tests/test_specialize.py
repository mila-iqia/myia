
from pytest import mark

from myia.api import parse, compile
from myia.prim.py_implementations import typeof, hastype, maplist, add
from myia.specialize import type_specialize, validate

from .test_infer import i64, f64


def specialize(*arglists):

    def decorate(fn):
        def run_test(args):
            arg_types = [{'type': typeof(arg)} for arg in args]

            g = parse(fn)
            result_py = fn(*args)
            result_orig = compile(g)(*args)
            assert result_py == result_orig
            g2 = type_specialize(g, arg_types)
            errs = validate(g2)
            if errs:
                print('Collected the following errors:')
                for node, e in errs.items():
                    print(f'   {node}')
                    print(f'      {" ".join(e)}')
                raise Exception('There are errors in the specialized graph.')
            result_final = compile(g2)(*args)
            assert result_py == result_final

        m = mark.parametrize('args', arglists)(run_test)
        m.__orig__ = fn
        return m

    return decorate


int1 = 13
int2 = 21

fp1 = 2.7
fp2 = 6.91


@specialize((int1, int2),
            (fp1, fp2))
def test_prim_mul(x, y):
    return x * y


@specialize((int1, int2),
            (fp1, int1))
def test_polymorphic(x, y):
    def helper(a, b):
        return a * a + b * b
    return helper(x, x + x), helper(y, y + y)


@specialize((int1, int2),
            (fp1, int1))
def test_polymorphic_closure(x, y):
    def construct(z):
        def inner(w):
            return z + w
        return inner
    return construct(x + x)(x), construct(y + y)(y)


@specialize((True, int1, int2),
            # (True, fp1, int1)  # TODO: mark this one as xfail
            )
def test_switch_fn(c, x, y):
    def dee(y):
        return y * y

    def doo(y):
        return y + y

    if c:
        f = dee
    else:
        f = doo

    return f(x), f(y)


@specialize((int1, int2), (int1, fp1))
def test_while(n, x):
    rval = x
    while n > 0:
        n = n - 1
        rval = rval - x
    return rval


@specialize((int1,), (fp1,))
def test_pow10(x):
    v = x
    j = 0
    while j < 3:
        i = 0
        while i < 3:
            v = v * x
            i = i + 1
        j = j + 1
    return v


@specialize((int1, fp1))
def test_hastype(x, y):
    def helper(x):
        if hastype(x, i64):
            return x
        elif hastype(x, f64):
            return x
        else:
            return (x,)

    return helper(x), helper(y), helper(())


@specialize(([fp1, fp2],))
def test_maplist(xs):
    def square(x):
        return x * x

    return maplist(square, xs)


@specialize(([fp1, fp2], [int1, int2]))
def test_maplist_polymorphic(xs, ys):
    def square(x):
        return x * x

    return maplist(square, xs), maplist(square, ys)


@mark.xfail(reason="Cannot specialize f")
@specialize((True, [fp1, fp2], [int1, int2]))
def test_maplist_polymorphic_2(c, xs, ys):
    def square(x):
        return x * x

    def double(x):
        return x + x

    if c:
        f = square
    else:
        f = double

    return maplist(f, xs), maplist(f, ys)


@specialize((int1, int2))
def test_unused_parameter(x, y):
    return x * x


@specialize((int1,))
def test_unused_function_parameter(x):
    # The type of square will be Dead(), but that's not really an issue
    # because it is indeed not used, and we can simply replace the reference
    # by a dummy.
    def square(x):
        return x * x

    def helper(f, a):
        return a * a
    return helper(square, x)


@specialize((int1,))
def test_indirect_primitive(x):
    def add2():
        return add

    return add2()(x, x)


@specialize((int1,))
def test_indirect_graph(x):
    def f(x):
        return x * x

    def f2():
        return f

    return f2()(x)
