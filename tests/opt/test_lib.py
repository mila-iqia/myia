
from .test_opt import _check_opt
from myia import dtype
from myia.composite import hyper_add
from myia.opt import lib
from myia.prim.py_implementations import \
    scalar_add, scalar_mul, tail, tuple_setitem, identity, partial, switch, \
    distribute, array_reduce, env_getitem, env_setitem, embed, env_add, \
    scalar_usub
from myia.utils import newenv


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


#######################
# Env simplifications #
#######################


def test_getitem_newenv():

    def before(x):
        return env_getitem(newenv, embed(x), 1234)

    def after(x):
        return 1234

    _check_opt(before, after,
               lib.getitem_newenv,
               argspec=[{'type': dtype.Float[64]}])


def test_env_get_set():

    def before(x, y):
        a = 5678
        e = env_setitem(newenv, embed(x), y)
        e = env_setitem(e, embed(a), a)
        return env_getitem(e, embed(x), 1234)

    def after(x, y):
        return y

    _check_opt(before, after,
               lib.cancel_env_set_get,
               argspec=[{'type': dtype.Float[64]},
                        {'type': dtype.Float[64]}])


def test_env_get_add():

    def before(x, y):
        e1 = env_setitem(newenv, embed(x), x)
        e1 = env_setitem(e1, embed(y), y)

        e2 = env_setitem(newenv, embed(y), y)
        e2 = env_setitem(e2, embed(x), x)

        return env_getitem(env_add(e1, e2), embed(x), 0)

    def after(x, y):
        return hyper_add(x, x)

    _check_opt(before, after,
               lib.getitem_env_add,
               lib.cancel_env_set_get,
               argspec=[{'type': dtype.Int[64]},
                        {'type': dtype.Int[64]}],
               argspec_after=False)


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


#######################
# Array optimizations #
#######################


def test_elim_distribute():

    def before(x):
        return distribute(x, (3, 5))

    def after(x):
        return x

    _check_opt(before, after,
               lib.elim_distribute,
               argspec=[{'type': dtype.Array[dtype.Float[64]],
                         'shape': (3, 5)}])

    _check_opt(before, before,
               lib.elim_distribute,
               argspec=[{'type': dtype.Array[dtype.Float[64]],
                         'shape': (3, 1)}])


def test_elim_array_reduce():

    def before(x):
        return array_reduce(scalar_add, x, (3, 1))

    def after(x):
        return x

    _check_opt(before, after,
               lib.elim_array_reduce,
               argspec=[{'type': dtype.Array[dtype.Float[64]],
                         'shape': (3, 1)}])

    _check_opt(before, before,
               lib.elim_array_reduce,
               argspec=[{'type': dtype.Array[dtype.Float[64]],
                         'shape': (3, 5)}])


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


def test_nested_switch_same_cond():

    def before(a, b, c, d):
        x = a < 0
        return switch(x, switch(x, a, b), switch(x, c, d))

    def after(a, b, c, d):
        x = a < 0
        return switch(x, a, d)

    _check_opt(before, after,
               lib.simplify_switch1,
               lib.simplify_switch2)


def test_nested_switch_diff_cond():

    def before(a, b, c, d):
        x = a < 0
        y = b < 0
        return switch(x, switch(y, a, b), switch(y, c, d))

    _check_opt(before, before,
               lib.simplify_switch1,
               lib.simplify_switch2)


def test_switch_same_branch():

    def before(x, y):
        a = y * y
        return switch(x < 0, a, a)

    def after(x, y):
        return y * y

    _check_opt(before, after,
               lib.simplify_switch_idem)


def test_combine_switch():

    def before(x, y):
        cond = x < 0
        a = switch(cond, x, y)
        b = switch(cond, y, x)
        return a + b

    def after(x, y):
        return switch(x < 0, x + y, y + x)

    _check_opt(before, after,
               lib.combine_switches)


def test_float_tuple_getitem_through_switch():

    def before(x, y):
        return switch(x < 0, x, y)[0]

    def after(x, y):
        return switch(x < 0, x[0], y[0])

    _check_opt(before, after,
               lib.float_tuple_getitem_through_switch)


def test_float_env_getitem_through_switch():

    def before(x, y):
        return env_getitem(switch(x < 0, newenv, newenv), embed(y), 5)

    def after(x, y):
        key = embed(y)
        return switch(x < 0,
                      env_getitem(newenv, key, 5),
                      env_getitem(newenv, key, 5))

    _check_opt(before, after,
               lib.float_env_getitem_through_switch)


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
# Specialization #
##################


def test_specialize_on_graph_arguments():

    def square(x):
        return x * x

    def before(x, y):
        def helper(f, x, g, y):
            return f(x) + g(y)

        return helper(square, x, scalar_usub, y)

    def after(x, y):
        def helper(x, y):
            return square(x) + scalar_usub(y)

        return helper(x, y)

    _check_opt(before, after,
               lib.specialize_on_graph_arguments)


#################
# Incorporation #
#################


def test_incorporate_getitem():

    def before(x, y):
        def b_help(x, y):
            return x * y, x + y
        return b_help(x, y)[0]

    def after(x, y):
        def a_help(x, y):
            return x * y
        return a_help(x, y)

    _check_opt(before, after,
               lib.incorporate_getitem)


def test_incorporate_getitem_2():

    def before(x, y):
        def b_help(x, y):
            return x
        return b_help(x, y)[0]

    def after(x, y):
        def a_help(x, y):
            return x[0]
        return a_help(x, y)

    _check_opt(before, after,
               lib.incorporate_getitem)


def test_incorporate_getitem_through_switch():

    def before(x, y):
        def f1(x, y):
            return x, y

        def f2(x, y):
            return y, x

        return switch(x < 0, f1, f2)(x, y)[0]

    def after(x, y):
        def f1(x, y):
            return x

        def f2(x, y):
            return y

        return switch(x < 0, f1, f2)(x, y)

    _check_opt(before, after,
               lib.incorporate_getitem_through_switch)


def test_incorporate_env_getitem():

    def before(x, y):
        key = embed(x)

        def b_help(x, y):
            return env_setitem(newenv, key, x * y)
        return env_getitem(b_help(x, y), key, 0)

    def after(x, y):
        def a_help(x, y):
            return x * y
        return a_help(x, y)

    _check_opt(before, after,
               lib.incorporate_env_getitem,
               lib.cancel_env_set_get,
               argspec=[{'type': dtype.Float[64]},
                        {'type': dtype.Float[64]}])


def test_incorporate_env_getitem_2():

    def before(x, y):
        def b_help(x, y):
            return x
        return env_getitem(b_help(x, y), 1234, 0)

    def after(x, y):
        def a_help(x, y):
            return env_getitem(x, 1234, 0)
        return a_help(x, y)

    _check_opt(before, after,
               lib.incorporate_env_getitem,
               lib.cancel_env_set_get)


def test_incorporate_env_getitem_through_switch():

    def before(x, y):
        key = embed(x)

        def f1(x, y):
            return env_setitem(newenv, key, x * y)

        def f2(x, y):
            return env_setitem(newenv, key, x + y)

        return env_getitem(switch(x < 0, f1, f2)(x, y), key, 0)

    def after(x, y):
        def f1(x, y):
            return x * y

        def f2(x, y):
            return x + y

        return switch(x < 0, f1, f2)(x, y)

    _check_opt(before, after,
               lib.incorporate_env_getitem_through_switch,
               lib.cancel_env_set_get,
               argspec=[{'type': dtype.Float[64]},
                        {'type': dtype.Float[64]}])


def test_incorporate_call():
    def b_help(q):
        def subf(z):
            return q * z
        return subf

    def before(x, y):
        return b_help(x)(y)

    def after(x, y):
        def a_help(q, y):
            def subf(z):
                return q * z
            return subf(y)
        return a_help(x, y)

    _check_opt(before, after,
               lib.incorporate_call)


def test_incorporate_call_through_switch():

    def before_helper(x):
        if x < 0:
            return scalar_mul
        else:
            return scalar_add

    def before(x, y, z):
        return before_helper(x)(y, z)

    def after(x, y, z):
        def after_helper(x, y, z):
            def tb(y, z):
                return scalar_mul(y, z)

            def fb(y, z):
                return scalar_add(y, z)

            return switch(x < 0, tb, fb)(y, z)

        return after_helper(x, y, z)

    _check_opt(before, after,
               lib.elim_identity,
               lib.incorporate_call,
               lib.incorporate_call_through_switch)
