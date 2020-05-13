from myia.abstract import DEAD
from myia.ir import isomorphic
from myia.operations import switch
from myia.opt import RemoveUnusedParameters
from myia.pipeline import scalar_parse, scalar_pipeline, steps

###################################
# Test removing unused parameters #
###################################


def step_rmunused(resources):
    while RemoveUnusedParameters(resources.opt_manager).run():
        pass


rmunused = scalar_pipeline.select(
    "resources",
    "parse",
    {"resolve": steps.step_resolve},
    {"rmunused": step_rmunused},
).make_transformer("input", "graph")


def test_rmunused_simple():
    @rmunused
    def f1(x, y):
        def g(z):
            return x

        return g(y)

    @scalar_parse
    def f2(x, y):
        def g():
            return x

        return g()

    assert isomorphic(f1, f2)


def test_rmunused_cascade():
    @rmunused
    def f1(x, y):
        def g(z):
            return h(x)

        def h(z):
            return x

        return g(y)

    @scalar_parse
    def f2(x, y):
        def g():
            return h()

        def h():
            return x

        return g()

    assert isomorphic(f1, f2)


def test_rmunused_middle():
    @rmunused
    def f1(x, y):
        def g(a, b, c):
            return a + c

        return g(x + 1, x + 2, x + 3)

    @scalar_parse
    def f2(x, y):
        def g(a, c):
            return a + c

        return g(x + 1, x + 3)

    assert isomorphic(f1, f2)


def test_rmunused_switch():
    @rmunused
    def f1(x, y):
        def g(a, b, c):
            return a + c

        def h(a, b, c):
            return a + c

        return switch(y < 0, g, h)(x + 1, x + 2, x + 3)

    @scalar_parse
    def f2(x, y):
        def g(a, c):
            return a + c

        def h(a, c):
            return a + c

        return switch(y < 0, g, h)(x + 1, x + 3)

    assert isomorphic(f1, f2)


def test_rmunused_switch_mustkeep():
    @rmunused
    def f1(x, y):
        def g(a, b, c):
            return a + c

        def h(a, b, c):
            return a + b

        return switch(y < 0, g, h)(x + 1, x + 2, x + 3)

    @scalar_parse
    def f2(x, y):
        def g(a, b, c):
            return a + c

        def h(a, b, c):
            return a + b

        return switch(y < 0, g, h)(x + 1, x + 2, x + 3)

    assert isomorphic(f1, f2)


def test_rmunused_switch_drop_one():
    @rmunused
    def f1(x, y):
        def g(a, b, c):
            return a

        def h(a, b, c):
            return c

        return switch(y < 0, g, h)(x + 1, x + 2, x + 3)

    @scalar_parse
    def f2(x, y):
        def g(a, c):
            return a

        def h(a, c):
            return c

        return switch(y < 0, g, h)(x + 1, x + 3)

    assert isomorphic(f1, f2)


def test_rmunused_switch_edge_case():
    def f(x, y):
        def g(a, b, c):
            return a

        def h(a, b, c):
            return a

        def hof(fn):
            return switch(y < 0, g, fn)(x + 1, x + 2, x + 3)

        return hof(h)

    # rmunused should do nothing
    f1 = rmunused(f)
    f2 = scalar_parse(f)

    assert isomorphic(f1, f2)


#######################
# Test lambda lifting #
#######################


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


def test_lift_switch():
    @llift
    def f1(x, y, z):
        if x < 0:
            return y
        else:
            return z

    @scalar_parse
    def f2(x, y, z):
        def true_branch(_y, _z):
            return _y

        def false_branch(_y, _z):
            return _z

        return switch(x < 0, true_branch, false_branch)(y, z)

    assert isomorphic(f1, f2)


def test_lift_switch2():
    @llift
    def f1(x, y, z):
        def g1():
            return y

        def g2():
            return z

        def g3():
            return 0

        a = switch(x < 0, g1, g3)
        b = switch(x > 0, g2, g3)
        return a() + b()

    @scalar_parse
    def f2(x, y, z):
        def g1(_z, _y):
            return _y

        def g2(_z, _y):
            return _z

        def g3(_z, _y):
            return 0

        a = switch(x < 0, g1, g3)
        b = switch(x > 0, g2, g3)
        return a(DEAD, y) + b(z, DEAD)

    assert isomorphic(f1, f2)
