from myia.parse import MyiaSyntaxError, parse_function, get_global_parse_env
from myia.inference.avm import \
    aroot_globals, abstract_evaluate, AbstractValue, ERROR, ANY
from myia.interpret import FunctionImpl
from myia.inference.types import *
from myia.symbols import builtins
import pytest


mark = pytest.mark
xfail = pytest.mark.xfail


_functions = {}


def val(x):
    return AbstractValue(x)


def iserror(av):
    print('error', av, av.values, ERROR in av.values)
    return ERROR in av.values


def getproj(av, proj):
    if proj in av.values:
        av = av[proj]
    if isinstance(av, AbstractValue) and ERROR in av.values:
        return False
    return av


def infer(**tests):

    tests = [(builtins[proj], t[:-1], t[-1])
             for proj, ts in tests.items()
             for t in ts]

    def decorate(fn):
        fname = fn.__name__
        fsym = get_global_parse_env(__file__).gen(fname, version=1)

        def test(proj, inputs, expected):
            if fsym not in _functions:
                _, genv = parse_function(fn)
                _functions.update(genv.bindings)
                for s, _lbda in genv.bindings.items():
                    aroot_globals[s] = FunctionImpl(_lbda, [aroot_globals])
            node = _functions[fsym]

            if not isinstance(inputs, tuple):
                inputs = inputs,
            if not isinstance(expected, set):
                expected = {expected}

            inputs = tuple(i if isinstance(i, AbstractValue)
                           else AbstractValue({proj: i})
                           for i in inputs)
            vm = abstract_evaluate(node, inputs, proj=proj)
            results = set(getproj(r, proj) for r in vm.result)
            assert results == expected

        m = pytest.mark.parametrize('proj,inputs,expected', list(tests))(test)
        m.__orig__ = fn
        return m

    return decorate


#################
# Trivial cases #
#################


@infer(type=[(Int8, Int8, Int8),
             (Int8, Int32, False),
             (Array[Float32], Array[Float32], Array[Float32]),
             (Array[Float32], Array[Float64], False),
             (Array[Float32], Bool, False)],
       shape=[((5, 6), (5, 6), (5, 6)),
              ((5, 3), (5, 9), False)])
def test_add(x, y):
    return x + y


@infer(shape=[((5, 6), (6, 10), (5, 10))])
def test_dot(x, y):
    return x @ y


@infer(type=[(val(-1), Float32, Float64, Float32),
             (val(1), Float32, Float64, Float64),
             (val(ANY), Float32, Float64, {Float32, Float64})],
       shape=[(val(-1), (5, 6), (7, 10), (5, 6)),
              (val(1), (5, 6), (7, 10), (7, 10)),
              (val(ANY), (5, 6), (7, 10), {(5, 6), (7, 10)})])
def test_if(sel, x, y):
    if sel < 0:
        return x + x
    else:
        return y + y


@infer(type=[(val(-1), Array[Float32], Array[Float32], Array[Float32]),
             # TODO: not sure how to express this one, because there is
             # a type error on the comparison, but the inferrer propagates
             # the return type anyway (which is not a bad thing).
             # (Array[Float32], Array[Float32], Array[Float32], False)
             ],
       shape=[(val(-1), (5, 6), (6, 10), (5, 10)),
              (val(1), (10, 6), (3, 10), (3, 6)),
              (val(ANY), (10, 6), (3, 10), {False, (3, 6)})])
def test_if2(sel, x, y):
    if sel < 0:
        a = x @ y
    else:
        a = y @ x
    return a


@infer(shape=[(val(3), (5, 6), (6, 10), (5, 10)),
              (val(ANY), (4, 6), (6, 12), (4, 12))])
def test_while(n, x, y):
    while n > 0:
        x = x + x
        n = n - 1
    return x @ y


@infer(shape=[((10, 3), (7, 3), (3, 7), (10, 7)),
              ((10, 3), (7, 3), (3, 6), False)])
def test_closures(a, b, y):
    def g(x):
        return x @ y
    return g(a) @ g(b)


@infer(shape=[((10, ANY), (ANY, 8), {(10, 8), False}),
              ((ANY, ANY), (ANY, ANY), {(ANY, ANY), False})])
def test_unknown(x, y):
    return x @ y


@infer(type=[(val(1_000_000), Array[Float64], Array[Float64], Array[Float64])],
       shape=[(val(0), (5, 6), (6, 8), (5, 6)),
              (val(1), (5, 6), (6, 8), (5, 8)),
              (val(2), (5, 6), (6, 8), False),
              (val(3), (5, 6), (6, 6), (5, 6)),
              # The following should **not** be executing 1,000,000 iterations
              (val(1_000_000), (5, 6), (6, 6), (5, 6)),
              (val(ANY), (5, 6), (6, 6), (5, 6)),
              (val(ANY), (5, 6), (6, 8), {False, (5, 6), (5, 8)})])
def test_precise_loop(n, x, y):
    while n > 0:
        x = x @ y
        n = n - 1
    return x


@xfail
@infer(shape=[(val(ANY), (5, 6), (10, 12), {(5, 6), (10, 12)})])
def test_precise_tracking(n, x, y):
    # The analysis thinks that this code can produce a shape
    # error because it does not understand that both occurrences
    # of f(x, y) must return the same value, even if n is unknown.
    def f(x, y):
        if n < 0:
            return x
        else:
            return y
    return f(x, y) + f(x, y)
