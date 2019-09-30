
from myia.lib import InferenceError
from myia.operations import hastype
from myia.pipeline import standard_debug_pipeline

from ..common import U, f64, i32, i64
from ..multitest import mt, run_debug
from ..test_grad import gradient
from ..test_infer import infer_scalar, infer_standard

_msg = 'There is no condition in which the program succeeds'


@mt(
    infer_scalar(i32, result=i32),
    infer_scalar(1, result=i64),
    infer_scalar(-1, result=InferenceError(_msg)),
)
def test_assert(x):
    assert x >= 0
    return x ** 0.5


@mt(
    infer_scalar(i32, result=i32),
    infer_scalar(1, result=i64),
    infer_scalar(-1, result=InferenceError(_msg)),
)
def test_assert_msg(x):
    assert x >= 0, 'x must be positive'
    return x ** 0.5


@mt(
    infer_scalar(i32, result=i32),

    run_debug(4.5, result=4.5**0.5),
    run_debug(-5, result=Exception),

    gradient(4.5, pipeline=standard_debug_pipeline.configure(validate=False)),
)
def test_raise(x):
    if x >= 0:
        return x ** 0.5
    else:
        raise Exception("sqrt of negative number")


@infer_scalar(i32, result=InferenceError(_msg))
def test_raise_unconditional(x):
    raise Exception("I don't like your face")


@infer_scalar(i32, result=i32)
def test_raise_multiple(x):
    if x < 0:
        raise Exception("You are too ugly")
    elif x > 0:
        raise Exception("You are too beautiful")
    else:
        return x


@mt(
    infer_standard(i32, result=InferenceError(_msg)),
    infer_standard(f64, result=f64),
    infer_standard(U(i32, i64), result=i64)
)
def test_raise_hastype(x):
    if hastype(x, i32):
        raise Exception("What a terrible type")
    else:
        return x


@infer_scalar(i32, result=i32)
def test_raise_loop(x):
    while x < 100:
        x = x * x
        if x > 150:
            raise Exception("oh no")
    return x


@infer_scalar(i32, result=i32)
def test_raise_rec(x):
    def f(x):
        if x == 0:
            return 1
        elif x >= 10:
            raise Exception("too big")
        else:
            return x * f(x - 1)
    return f(x)
