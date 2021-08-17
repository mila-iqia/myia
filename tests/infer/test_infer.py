import operator

from myia.abstract.map import MapError
from myia.abstract.to_abstract import to_abstract, type_to_abstract
from myia.infer.infnode import infer_graph, inferrers, signature
from myia.parser import parse
from myia.testing.common import Nil, Ty
from myia.testing.multitest import infer, mt
from myia.utils.info import enable_debug


def f(x, y):
    # Note: test_specialization overrides / to have type (int, float) -> float
    return g(x) / g(y)


def g(z):
    return -z


def test_specialization():
    # TODO: This line seems to break str() on representations in other tests,
    # to reproduce, call signature with these arguments before all tests are run
    inferrers[operator.truediv] = signature(int, float, ret=float)

    with enable_debug():
        graph = parse(f)

    g = infer_graph(graph, (to_abstract(1), to_abstract(1.5)))
    result = g.return_.abstract
    assert result is type_to_abstract(float)

    # TODO: verify that all of g's nodes have a type


# Test `infer` as a decorator
@infer(int, int, result=int)
def test_infer_decorator(a, b):
    return a + b


@mt(
    infer(int, int, result=int),
    infer(float, float, result=float),
    infer(int, float, result=float),
    # This is for coverage of missing builtin function, it could be
    # a different builtin when we add support for list
    infer(list, list, result=TypeError),
)
def test_sum_method(a, b):
    return a.__add__(b)


def fact(x):
    if x <= 1:
        return x
    else:
        return x * fact(x - 1)


@mt(
    infer(int, result=int),
    infer(float, result=float),
)
def test_fact(x):
    return fact(x)


@infer(int, result=TypeError)
def test_not_function(x):
    return x()


@infer(result=int)
def test_nullary_call():
    def f():
        return 1

    return f()


@infer(int, result=int)
def test_constant_branch(x):
    if x <= 0:
        return 1
    else:
        return 2


@infer(int, result=int)
def test_module_function_call(x):
    return operator.neg(x)


@mt(
    # we could not cast to a Nil,
    infer(Ty(Nil), int, result=Exception("wrong number of arguments")),
    infer(Ty(bool), bool, result=bool),
    infer(Ty(bool), int, result=bool),
    infer(Ty(bool), float, result=bool),
    infer(Ty(int), bool, result=int),
    infer(Ty(int), int, result=int),
    infer(Ty(int), float, result=int),
    infer(Ty(float), bool, result=float),
    infer(Ty(float), int, result=float),
    infer(Ty(float), float, result=float),
)
def test_infer_scalar_cast(dtype, value):
    return dtype(value)


class Banana:
    def __init__(self, size):
        self.size = size

    def bigger(self, obj):
        return obj <= self.size


@mt(
    infer(Banana(3.5), float, result=bool),
    infer(Banana("ohno"), float, result=MapError),
)
def test_method(b, x):
    return b.bigger(x)


##########################
# Test builtin operators #
##########################


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_add(a, b):
    return a + b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_floordiv(a, b):
    return a // b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_mod(a, b):
    return a % b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_mul(a, b):
    return a * b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_pow(a, b):
    return a ** b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=float),
)
def test_binary_sub(a, b):
    return a - b


@mt(
    infer(int, bool, result=float),
    infer(int, int, result=float),
    infer(int, float, result=float),
    infer(float, bool, result=float),
    infer(float, int, result=float),
    infer(float, float, result=float),
    infer(bool, bool, result=float),
    infer(bool, int, result=float),
    infer(bool, float, result=float),
)
def test_binary_truediv(a, b):
    return a / b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_eq(a, b):
    return a == b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_ge(a, b):
    return a >= b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_gt(a, b):
    return a > b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_is(a, b):
    return a is b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_is_not(a, b):
    return a is not b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_le(a, b):
    return a <= b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_lt(a, b):
    return a < b


@mt(
    infer(int, bool, result=bool),
    infer(int, int, result=bool),
    infer(int, float, result=bool),
    infer(float, bool, result=bool),
    infer(float, int, result=bool),
    infer(float, float, result=bool),
    infer(bool, bool, result=bool),
    infer(bool, int, result=bool),
    infer(bool, float, result=bool),
)
def test_binary_bool_ne(a, b):
    return a != b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=AssertionError("No inference")),
    infer(
        float,
        bool,
        result=TypeError(
            "No inferrer for <slot wrapper '__rlshift__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        int,
        result=TypeError(
            "No inferrer for <slot wrapper '__rlshift__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        float,
        result=TypeError(
            "No __lshift__ method for <class 'float'> and no __rlshift__ method for <class 'float'>"
        ),
    ),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=AssertionError("No inference")),
)
def test_bitwise_binary_lshift(a, b):
    return a << b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=AssertionError("No inference")),
    infer(
        float,
        bool,
        result=TypeError(
            "No inferrer for <slot wrapper '__rrshift__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        int,
        result=TypeError(
            "No inferrer for <slot wrapper '__rrshift__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        float,
        result=TypeError(
            "No __rshift__ method for <class 'float'> and no __rrshift__ method for <class 'float'>"
        ),
    ),
    infer(bool, bool, result=int),
    infer(bool, int, result=int),
    infer(bool, float, result=AssertionError("No inference")),
)
def test_bitwise_binary_rshift(a, b):
    return a >> b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=AssertionError("No inference")),
    infer(
        float,
        bool,
        result=TypeError(
            "No inferrer for <slot wrapper '__rand__' of 'bool' objects>"
        ),
    ),
    infer(
        float,
        int,
        result=TypeError(
            "No inferrer for <slot wrapper '__rand__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        float,
        result=TypeError(
            "No __and__ method for <class 'float'> and no __rand__ method for <class 'float'>"
        ),
    ),
    infer(bool, bool, result=bool),
    infer(bool, int, result=int),
    infer(bool, float, result=AssertionError("No inference")),
)
def test_bitwise_binary_and(a, b):
    return a & b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=AssertionError("No inference")),
    infer(
        float,
        bool,
        result=TypeError(
            "No inferrer for <slot wrapper '__ror__' of 'bool' objects>"
        ),
    ),
    infer(
        float,
        int,
        result=TypeError(
            "No inferrer for <slot wrapper '__ror__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        float,
        result=TypeError(
            "No __or__ method for <class 'float'> and no __ror__ method for <class 'float'>"
        ),
    ),
    infer(bool, bool, result=bool),
    infer(bool, int, result=int),
    infer(bool, float, result=AssertionError("No inference")),
)
def test_bitwise_binary_or(a, b):
    return a | b


@mt(
    infer(int, bool, result=int),
    infer(int, int, result=int),
    infer(int, float, result=AssertionError("No inference")),
    infer(
        float,
        bool,
        result=TypeError(
            "No inferrer for <slot wrapper '__rxor__' of 'bool' objects>"
        ),
    ),
    infer(
        float,
        int,
        result=TypeError(
            "No inferrer for <slot wrapper '__rxor__' of 'int' objects>"
        ),
    ),
    infer(
        float,
        float,
        result=TypeError(
            "No __xor__ method for <class 'float'> and no __rxor__ method for <class 'float'>"
        ),
    ),
    infer(bool, bool, result=bool),
    infer(bool, int, result=int),
    infer(bool, float, result=AssertionError("No inference")),
)
def test_bitwise_binary_xor(a, b):
    return a ^ b


@mt(
    infer(int, result=int),
    infer(bool, result=int),
    infer(
        float, result=TypeError("No __invert__ method for type <class 'float'>")
    ),
)
def test_bitwise_unary_invert(a):
    return ~a


@mt(
    infer(int, result=int),
    infer(bool, result=int),
    infer(float, result=float),
)
def test_unary_pos(a):
    return +a


@mt(
    infer(int, result=int),
    infer(bool, result=bool),
    infer(float, result=float),
)
def test_unary_neg(a):
    return -a


@mt(
    infer(int, result=bool),
    infer(bool, result=bool),
    infer(float, result=bool),
)
def test_unary_not(a):
    return not a
