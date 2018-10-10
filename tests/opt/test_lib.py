
from pytest import mark

from .test_opt import _check_opt
from myia.opt import lib
from myia.prim.py_implementations import \
    tail, tuple_setitem, scalar_add, scalar_mul, identity, partial, switch


#######################
# Tuple optimizations #
#######################


def test_getitem_tuple_elem0():

    def before(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[0]

    def after(x):
        return x + 1

    _check_opt(before, after,
               lib.getitem_tuple)


def test_getitem_tuple_elem3():

    def before1(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[3]

    def before2(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tail(tail(tail(tup)))[0]

    def after(x):
        return x + 4

    _check_opt(before1, after,
               lib.getitem_tuple)

    _check_opt(before2, after,
               lib.getitem_tuple, lib.tail_tuple)


def test_getitem_tuple_noopt():

    def before1(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[y]

    def before2(x):
        return tail(x)

    _check_opt(before1, before1,
               lib.getitem_tuple)

    _check_opt(before2, before2,
               lib.tail_tuple)


def test_setitem_tuple_elem0():

    def before(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tuple_setitem(tup, 0, y)

    def after(x, y):
        tup = (y, x + 2, x + 3, x + 4)
        return tup

    _check_opt(before, after,
               lib.setitem_tuple)


def test_setitem_tuple_elem3():

    def before(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tuple_setitem(tup, 3, y)

    def after(x, y):
        tup = (x + 1, x + 2, x + 3, y)
        return tup

    _check_opt(before, after,
               lib.setitem_tuple)


def test_setitem_tuple_noopt():

    def before(x, y, z):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tuple_setitem(tup, z, y)

    _check_opt(before, before,
               lib.setitem_tuple)


def test_op_tuple_binary():

    def before(x, y, z):
        return (x, y, z) + (1, 2, 3)

    def after(x, y, z):
        return (x + 1, y + 2, z + 3)

    _check_opt(before, after,
               lib.bubble_op_tuple_binary)


##############################
# Arithmetic simplifications #
##############################


def test_mul_zero():

    def before1(x):
        return x * 0

    def before2(x):
        return 0 * x

    def after(x):
        return 0

    _check_opt(before1, after,
               lib.multiply_by_zero_l,
               lib.multiply_by_zero_r)

    _check_opt(before2, after,
               lib.multiply_by_zero_l,
               lib.multiply_by_zero_r)


def test_mul_one():

    def before1(x):
        return x * 1

    def before2(x):
        return 1 * x

    def after(x):
        return x

    _check_opt(before1, after,
               lib.multiply_by_one_l,
               lib.multiply_by_one_r)

    _check_opt(before2, after,
               lib.multiply_by_one_l,
               lib.multiply_by_one_r)


def test_add_zero():

    def before1(x):
        return x + 0

    def before2(x):
        return 0 + x

    def after(x):
        return x

    _check_opt(before1, after,
               lib.add_zero_l,
               lib.add_zero_r)

    _check_opt(before2, after,
               lib.add_zero_l,
               lib.add_zero_r)


def test_elim_identity():

    def before(x, y):
        return identity(x) + identity(y)

    def after(x, y):
        return x + y

    _check_opt(before, after, lib.elim_identity)


######################
# Branch elimination #
######################


def test_true_branch():

    def before(x, y):
        if True:
            return x
        else:
            return y

    def after(x, y):
        return x

    _check_opt(before, after,
               lib.elim_identity,
               lib.simplify_always_true,
               lib.inline)


def test_false_branch():

    def before(x, y):
        if False:
            return x
        else:
            return y

    def after(x, y):
        return y

    _check_opt(before, after,
               lib.elim_identity,
               lib.simplify_always_false,
               lib.inline)


def test_true_branch_switch():

    def before(x, y):
        return switch(True, x, y)

    def after(x, y):
        return x

    _check_opt(before, after,
               lib.simplify_always_true)


def test_false_branch_switch():

    def before(x, y):
        return switch(False, x, y)

    def after(x, y):
        return y

    _check_opt(before, after,
               lib.simplify_always_false)


############
# Partials #
############


def test_partials():

    def f(x, y):
        return x + y

    def before(x, y):
        return partial(f, x)(y)

    def after(x, y):
        return f(x, y)

    _check_opt(before, after,
               lib.simplify_partial)


############
# Inlining #
############


def test_inline():

    def f(x, y):
        return x + y

    def before(x, y, z):
        return f(x * y, y * z)

    def after(x, y, z):
        return x * y + y * z

    _check_opt(before, after,
               lib.inline)


def test_inline_successively():

    def one(x):
        return x + 1

    def two(x):
        return one(x + 2)

    def three(x):
        return two(x + 3)

    def before(x):
        return three(x)

    def after(x):
        return x + 3 + 2 + 1

    _check_opt(before, after,
               lib.inline)


def test_inline_closure():

    def before(x, y, z):
        c = z * z

        def f(x):
            return x + c
        return f(x * y)

    def after(x, y, z):
        c = z * z
        return x * y + c

    _check_opt(before, after,
               lib.inline)


def test_inline_deep_closure():

    def f(x):
        w = x * x

        def g():
            def h():
                return w
            return h()
        return g

    def before(x, y):
        return f(x)() - f(y)()

    def after(x, y):
        w1 = x * x
        w2 = y * y
        return w1 - w2

    _check_opt(before, after,
               lib.inline)


def test_inline_new_closure():

    def q(x):
        return x * x

    def f(x):
        def g():
            return q(x)
        return g

    def before(x):
        return f(x)

    def after(x):
        def g():
            return x * x
        return g

    _check_opt(before, after,
               lib.inline)


def test_inline_recursive_direct():
    # Optimizations will not apply in order not to loop infinitely

    def before1(x):
        return before1(x - 1)

    def helper2(x):
        return before2(x - 1)

    def before2(x):
        return helper2(x - 1)

    _check_opt(before1, before1,
               lib.inline)

    _check_opt(before2, before2,
               lib.inline)


def test_inline_recursive():

    def before(x):
        if x <= 0:
            return x
        else:
            return before(x - 1)

    _check_opt(before, before,
               lib.inline)


def test_inline_criterion():

    inline_binary = lib.make_inliner(
        lambda node, g, args: len(args) == 2,
        check_recursive=False
    )

    def bin(x, y):
        return x * y

    def un(x):
        return x - 3

    def before(x, y, z):
        return un(x) + bin(y, z)

    def after(x, y, z):
        return un(x) + y * z

    _check_opt(before, after,
               inline_binary)


def test_inline_trivial():

    def trivial(x):
        return x * x

    def nontrivial(x):
        return x * x * x

    def before(x):
        return trivial(x), nontrivial(x)

    def after(x):
        return x * x, nontrivial(x)

    _check_opt(before, after,
               lib.inline_trivial)


@mark.xfail(reason="inline_trivial does not look into closures properly")
def test_inline_nontrivial_through_fv():

    def nontrivial(x):
        z = (x * x) * (x * x)

        def g():
            return z
        return g

    def before(x, y):
        return nontrivial(x), nontrivial(y)

    def after_all_inline(x, y):
        z1 = (x * x) * (x * x)
        z2 = (y * y) * (y * y)

        def g1():
            return z1

        def g2():
            return z2
        return g1, g2

    # Inlining everything bloats the graph
    _check_opt(before, after_all_inline,
               lib.inline)

    # inline_trivial should avoid this
    _check_opt(before, before,
               lib.inline_trivial)


def test_inline_unique_uses():

    def one(x):
        return x * x

    def two(x):
        return x + x

    def before(x):
        return one(x), two(x), two(x)

    def after(x):
        return x * x, two(x), two(x)

    _check_opt(before, after,
               lib.inline_unique_uses)


def test_inline_unique_uses_2():

    def f(x):
        return x * x

    def g(x):
        return f(x)

    def h(x):
        return f(x)

    def before(x):
        return g(x) + h(x)

    def after(x):
        return f(x) + f(x)

    _check_opt(before, after,
               lib.inline_unique_uses)


def test_inline_unique_uses_recursive():

    def helper(x):
        return before(x)

    def before(x):
        return helper(x)

    _check_opt(before, before,
               lib.inline_unique_uses)


def test_replace_applicator():

    def app1(x, y):
        return x + y

    def app2(x, y):
        return app1(x, y)

    def app3(x, y):
        return y + x

    def before1(x, y):
        return app1(x, y)

    def before2(x, y):
        return app2(x, y)

    def before3(x, y):
        return app3(x, y)

    def after(x, y):
        return x + y

    _check_opt(before1, after,
               lib.replace_applicator)

    _check_opt(before2, after,
               lib.replace_applicator)

    _check_opt(before3, before3,
               lib.replace_applicator)


def test_replace_applicator_2():

    def before(x, y):
        def app(x, y):
            z = x * y

            def inner(x, y):
                return z
            return inner(x, y)
        return app(x + 1, y + 1)

    _check_opt(before, before,
               lib.replace_applicator)


##################
# Drop call into #
##################


def test_drop_into():
    def b_help(q):
        def subf(z):
            return q * z
        return subf

    def before(x, y):
        return b_help(x)(y)

    def after(x, y):
        def a_help(q):
            def subf(z):
                return q * z
            return subf(y)
        return a_help(x)

    _check_opt(before, after,
               lib.drop_into_call)
