from myia.api import parse, run


def parse_compare(args):
    def wrap(fn):
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
def test_simple5(x):
    y = x + 1
    z = x + 2
    w = y + z
    return w + x + y + z


@parse_compare([2])
def test_fn1(x):
    def g(x):
        return x
    return g(x)


@parse_compare([])
def test_fn2():
    def g(x):
        def f():
            return x
        return f()
    return g(2)


@parse_compare([])
def test_fn3():
    def g(x):
        def f():
            return x
        return f() + 1
    return g(2)


@parse_compare([])
def test_fn4():
    def g(x):
        y = x + 1

        def f():
            return y
        return f() + 1
    return g(2)


@parse_compare([])
def test_fn5():
    def g(x):
        def f(y):
            return y + 1
        return f(x + 1)
    return g(2)


@parse_compare([])
def test_closure():
    def g(x):
        def f():
            return x
        return f
    return g(2)()


@parse_compare([0])
def test_if1(x):
    if x:
        y = 1
    else:
        y = 0
    return y


@parse_compare([2])
def test_if2(x):
    if x:
        y = 1
    else:
        y = 0
    return y
