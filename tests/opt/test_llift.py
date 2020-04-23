from myia.ir import isomorphic
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
