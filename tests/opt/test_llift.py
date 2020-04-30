from myia.ir import isomorphic
from myia.operations import switch
from myia.pipeline import scalar_parse, scalar_pipeline, steps

llift = scalar_pipeline.select(
    "resources", "parse", {"resolve": steps.step_resolve}, "llift",
).make_transformer("input", "graph")


def test_lambda_lift_simple():
    @llift
    def f1(x, y):
        def g(z):
            return x + z

        return g(y)

    @scalar_parse
    def f2(x, y):
        def g(z, _x):
            return _x + z

        return g(y, x)

    assert isomorphic(f1, f2)


def test_lambda_lift_nested():
    @llift
    def f1(x, y):
        def g(z):
            def h():
                return x + z

            return h()

        return g(y)

    @scalar_parse
    def f2(x, y):
        def g(z, _x):
            def h(__x, _z):
                return __x + _z

            return h(_x, z)

        return g(y, x)

    assert isomorphic(f1, f2)


def test_lambda_lift_chain():
    @llift
    def f1(x, y):
        def g(z):
            return x + z

        def h():
            return g(y)

        return h()

    @scalar_parse
    def f2(x, y):
        def g(z, _x):
            return _x + z

        def h(_y, _x):
            return g(_y, _x)

        return h(y, x)

    assert isomorphic(f1, f2)


def test_lambda_change_nesting():
    # Originally g and h are not nested in i, and since they cannot be lambda
    # lifted because they are not in call position, they must be moved into the
    # scope of i so that they can point to i's new parameter instead of the top
    # level one.

    @llift
    def f1(x):
        def g():
            return x

        def h():
            return -x

        def i():
            return switch(x > 0, g, h)

        return i()()

    @scalar_parse
    def f2(x):
        def i(_x):
            def g():
                return _x

            def h():
                return -_x

            return switch(_x > 0, g, h)

        return i(x)()

    assert isomorphic(f1, f2)


def test_cannot_lambda_lift():
    # Cannot lambda_lift because g is not exclusively used in call position.

    def f(x, y):
        def g(z):
            return x + z

        def h(fn, arg):
            return fn(arg)

        return g(y) + h(g, x)

    # llift should do nothing
    f1 = llift(f)
    f2 = scalar_parse(f)

    assert isomorphic(f1, f2)
