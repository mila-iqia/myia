
import pytest
from .test_opt import _check_opt
from myia.opt import lib
from myia.prim import ops
from myia.prim.py_implementations import head, tail, setitem


#######################
# Tuple optimizations #
#######################


def test_getitem_tuple_elem0():

    def before1(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[0]

    def before2(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return head(tup)

    def after(x):
        return x + 1

    _check_opt(before1, after,
               lib.getitem_tuple)

    _check_opt(before2, after,
               lib.head_tuple)


def test_getitem_tuple_elem3():

    def before1(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[3]

    def before2(x):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return head(tail(tail(tail(tup))))

    def after(x):
        return x + 4

    _check_opt(before1, after,
               lib.getitem_tuple)

    _check_opt(before2, after,
               lib.head_tuple, lib.tail_tuple)


def test_getitem_tuple_noopt():

    def before1(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return tup[y]

    def before2(x):
        return head(x)

    def before3(x):
        return tail(x)

    _check_opt(before1, before1,
               lib.getitem_tuple)

    _check_opt(before2, before2,
               lib.head_tuple)

    _check_opt(before3, before3,
               lib.tail_tuple)


def test_setitem_tuple_elem0():

    def before(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return setitem(tup, 0, y)

    def after(x, y):
        tup = (y, x + 2, x + 3, x + 4)
        return tup

    _check_opt(before, after,
               lib.setitem_tuple)


def test_setitem_tuple_elem3():

    def before(x, y):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return setitem(tup, 3, y)

    def after(x, y):
        tup = (x + 1, x + 2, x + 3, y)
        return tup

    _check_opt(before, after,
               lib.setitem_tuple)


def test_setitem_tuple_noopt():

    def before(x, y, z):
        tup = (x + 1, x + 2, x + 3, x + 4)
        return setitem(tup, z, y)

    _check_opt(before, before,
               lib.setitem_tuple)


def test_op_tuple_binary():

    def before(x, y, z):
        return (x, y, z) + (1, 2, 3)

    def after(x, y, z):
        return (x + 1, y + 2, z + 3)

    _check_opt(before, after,
               lib.bubble_op_cons_binary,
               lib.bubble_op_nil_binary)


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


########################
# Constant propagation #
########################


def test_ctprop():

    def before(x):
        return 10 + 5 * 8

    def after(x):
        return 50

    _check_opt(before, after,
               lib.constant_prop)


def test_ctprop2():

    def before(x):
        return (6 - 3) + x * (8 * 2)

    def after(x):
        return 3 + x * 16

    _check_opt(before, after,
               lib.constant_prop)


@pytest.mark.xfail(reason="Should evaluate helper.")
def test_ctprop_helper():

    def helper(x):
        return x * x

    def before(x):
        return helper(3) + x

    def after(x):
        return 9 + x

    _check_opt(before, after,
               lib.constant_prop)


def test_ctprop_closure():

    def before(x):
        def helper(y):
            return x * y
        return helper(3) + x

    _check_opt(before, before,
               lib.constant_prop)


def test_ctprop_subset():
    prop = lib.make_constant_prop({
        ops.mul: lambda x, y: x * y
    })

    def before():
        return (5 * 8) + 3

    def after():
        return 40 + 3

    _check_opt(before, after,
               prop)


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
