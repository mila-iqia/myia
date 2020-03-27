from myia.lib import from_value
from myia.operations import scalar_mul, tagged
from myia.pipeline import standard_pipeline

from .common import Pair, f64, i64
from .multitest import backend_all, mt, run, run_debug
from .test_grad import gradient
from .test_infer import infer_scalar
from .test_monomorphize import mono_scalar

#########################
# Arithmetic algorithms #
#########################


@mt(
    infer_scalar(i64, result=i64),
    infer_scalar(f64, result=f64),
    run(3),
    run(4.8),
    gradient(2.0),
)
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


@mt(
    infer_scalar(i64, result=i64),
    infer_scalar(f64, result=f64),
    run_debug(0),
    run_debug(1),
    run_debug(4),
    run(10),
    gradient(4.1),
)
def test_fact(n):
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    return fact(n)


@mt(run(8), mono_scalar(13))  # Covers an edge case in monomorphize
def test_fib(n):
    a = 1
    b = 1
    for _ in range(n):
        a, b = b, a + b
    return a


#######################
# Algorithms on trees #
#######################


def make_tree(depth, x):
    if depth == 0:
        return tagged(x)
    else:
        return tagged(
            Pair(make_tree(depth - 1, x * 2), make_tree(depth - 1, x * 2 + 1))
        )


def countdown(n):
    if n == 0:
        return tagged(None)
    else:
        return tagged(Pair(n, countdown(n - 1)))


def sumtree(t):
    if isinstance(t, (int, float)):
        return t
    elif t is None:
        return 0
    else:
        return sumtree(t.left) + sumtree(t.right)


def reducetree(fn, t, init):
    if isinstance(t, (int, float)):
        return t
    elif t is None:
        return init
    else:
        return fn(reducetree(fn, t.left, init), reducetree(fn, t.right, init))


pair_t1 = from_value(Pair(Pair(1, 2), Pair(2, 3)))
pair_t1_u = pair_t1.attributes["left"]


@infer_scalar(i64, result=pair_t1_u)
def test_make_tree(depth):
    return make_tree(depth, 1)


pair_t2 = from_value(Pair(1, Pair(2, Pair(3, None))))
pair_t2_u = pair_t2.attributes["right"]


@infer_scalar(i64, result=pair_t2_u)
def test_countdown(depth):
    return countdown(depth)


@mt(
    run(make_tree(3, 1)),
    run(countdown(10)),
    gradient(make_tree(3, 1.0)),
    gradient(countdown(3.0)),
    pipeline=standard_pipeline,
    backend=backend_all,
)
def test_sumtree(x):
    return sumtree(x)


@mt(
    run(make_tree(3, 1), 1),
    run(countdown(10), 1),
    gradient(make_tree(3, 1.0), 1.0),
    gradient(countdown(4.0), 1.0),
    pipeline=standard_pipeline,
    backend=backend_all,
)
def test_reducetree(t, init):
    return reducetree(scalar_mul, t, init)
