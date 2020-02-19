
from myia.operations import einsum
from myia.lib import InferenceError

from ..multitest import infer, mt, run_debug
from ..common import MA, MB


@run_debug(MA(3, 3)[0], MA(3, 3)[1])
def test_einsum_inner(a, b):
    return einsum('j,j', a, b)


@run_debug(MA(3, 3)[0], MA(3, 3)[1])
def test_einsum_outer(a, b):
    return einsum('i,k->ik', a, b)


@run_debug(MA(2, 4))
def test_einsum_transpose(a):
    return einsum('ij->ji', a)


@mt(
    run_debug(MA(2, 3), MB(3, 4)),
    infer(MA(2, 3), MB(4, 3), result=InferenceError),
    infer(MA(2, 3), MB(3, 4)[1], result=InferenceError),
    infer(MA(2, 3), MB(3, 4), MA(2, 2), result=InferenceError)
)
def test_einsum_mm(a, b):
    return einsum('ij,jk->ik', a, b)


@run_debug(MA(6, 4).reshape(2, 3, 4))
def test_einsum_red(a):
    return einsum('ijk->ik', a)
