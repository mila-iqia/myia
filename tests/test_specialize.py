
from pytest import mark
from collections import defaultdict

from myia.api import standard_pipeline, step_debug_export
from myia.dtype import Function, Unknown, Type, Problem, External
from myia.graph_utils import dfs
from myia.infer import Inferrer
from myia.ir import succ_deeper, is_apply, is_constant
from myia.prim import ops as P, Primitive
from myia.prim.py_implementations import typeof, hastype, maplist, add
from myia.specialize import DEAD

from .test_infer import i64, f64


specialize_pipeline = standard_pipeline.select(
    'parse', 'infer', 'specialize'
).insert_after(
    debug_export=step_debug_export
).configure(
    {'infer.tracks.value.max_depth': 1}
)


# Add ops as needed, but don't add getattr, resolve and other ops that we want
# to eliminate at this stage.
op_whitelist = [
    P.return_, P.if_, P.partial,
    P.add, P.mul, P.sub,
    P.gt, P.lt,
    P.cons_tuple, P.hastype, P.maplist
]


def validate(g):
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """
    errors = defaultdict(set)
    for node in dfs(g.return_, succ_deeper):
        if node.type is None or node.type == Unknown():
            errors[node].add('No type')
        elif isinstance(node.type, Inferrer):
            errors[node].add('Uneliminated inferrer')
        elif isinstance(node.type, Problem):
            if node.type.kind is DEAD:
                # This one is okay if it happens, because we don't really need
                # to infer types for dead code.
                pass
            else:
                errors[node].add(f'Problem type: {node.type}')
        elif isinstance(node.type, External):
            errors[node].add(f'External type: {node.type}')
        elif not isinstance(node.type, Type):
            errors[node].add(f'Unknown type: {node.type}')
        elif is_apply(node):
            expected = Function([i.type for i in node.inputs[1:]], node.type)
            if node.inputs[0].type != expected:
                errors[node].add('Function/argument inconsistency')
            fn = node.inputs[0]
            if is_constant(fn, Primitive) and fn.value not in op_whitelist:
                errors[node].add(f'Forbidden primitive: {fn.value}')

    return errors


def specialize(*arglists):

    def decorate(fn):
        def run_test(args):
            arg_types = [{'type': typeof(arg)} for arg in args]

            result_py = fn(*args)

            res = specialize_pipeline.make()(input=fn, argspec=arg_types)
            g2 = res['graph']

            errs = validate(g2)
            if errs:
                print('Collected the following errors:')
                for node, e in errs.items():
                    print(f'   {node}')
                    print(f'      {" ".join(e)}')
                raise Exception('There are errors in the specialized graph.')
            result_final = res['output'](*args)
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
    # The type of square will be Problem(DEAD), but that's not really an issue
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


@specialize((True, int1, int1))
def test_poly_with_constants(c, x, y):
    def f1(x, y):
        return x + y

    def f2(x, y):
        return y + x

    def choose(c):
        if c:
            return f1
        else:
            return f2

    return choose(c)(x, 2), choose(not c)(2, y)


@specialize((int1, int2), (fp1, fp2))
def test_method(x, y):
    return x.__add__(y)


@specialize((int1, fp1))
def test_method_polymorphic(x, y):
    return x.__add__(x), y.__add__(y)
