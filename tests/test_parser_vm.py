import functools
from myia.api import parse, run

def parse_compare(args):
    def wrap(fn):
        # @functools.wraps(fn)
        def test():
            ref = fn(*args)
            res = run(parse(fn), args)
            assert ref == res
        return test
    return wrap


@parse_compare([])
def test_simple1():
    return 1


@parse_compare([2])
def test_simple2(x):
    return x


@parse_compare([1])
def test_simple3(x):
    return x + 1


@parse_compare([2, 3])
def test_simple4(x, y):
    return x + y


@parse_compare([2])
def test_fn1(x):
    def g(x):
        return x
    y = g(x)
    return y
