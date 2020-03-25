import pytest

from myia.lib import InferenceError
from myia.operations import einsum

from ..common import MA, MB
from ..multitest import infer, mt, run_debug


@run_debug(MA(1, 4)[0])
def test_einsum_view1d(a):
    return einsum("i", a)


@run_debug(MA(1, 4)[0])
def test_einsum_sum1d(a):
    return einsum("i->", a)


@run_debug(MA(4, 1)[0], MB(4, 1)[0])
def test_einsum_elemwise1d(a, b):
    return einsum("i,i->i", a, b)


@run_debug(MA(3, 3)[0], MA(3, 3)[1])
def test_einsum_inner(a, b):
    return einsum("j,j", a, b)


@run_debug(MA(3, 3)[0], MA(3, 3)[1])
def test_einsum_outer(a, b):
    return einsum("i,k->ik", a, b)


@run_debug(MA(2, 4))
def test_einsum_view2d(a):
    return einsum("ij->ij", a)


@run_debug(MA(2, 4))
def test_einsum_transpose(a):
    return einsum("ij->ji", a)


@pytest.mark.xfail
@run_debug(MA(4, 4))
def test_einsum_diag(a):
    return einsum("ii->i", a)


@pytest.mark.xfail
@run_debug(MA(4, 4))
def test_einsum_trace(a):
    return einsum("ii", a)


@run_debug(MA(3, 4))
def test_einsum_sum2d_0(a):
    return einsum("ij->i", a)


@run_debug(MA(3, 4))
def test_einsum_sum2d_1(a):
    return einsum("ij->j", a)


@run_debug(MA(3, 4))
def test_einsum_sum2d_01(a):
    return einsum("ij->", a)


@run_debug(MA(3, 4), MB(3, 4))
def test_einsum_elemwise2d(a, b):
    return einsum("ij,ij->ij", a, b)


@run_debug(MA(3, 4), MB(4, 3))
def test_einsum_elemwise2d_T(a, b):
    return einsum("ij,ji->ij", a, b)


@mt(
    run_debug(MA(2, 3), MB(3, 4)),
    infer(MA(2, 3), MB(4, 3), result=InferenceError),
    infer(MA(2, 3), MB(3, 4)[1], result=InferenceError),
    infer(MA(2, 3), MB(3, 4), MA(2, 2), result=InferenceError),
)
def test_einsum_mm(a, b):
    return einsum("ij,jk->ik", a, b)


@run_debug(MA(2, 3), MB(4, 3))
def test_einsum_inner2d(a, b):
    return einsum("ij,kj->ik", a, b)


@run_debug(MA(2, 3), MB(4, 3))
def test_einsum_rowmul(a, b):
    return einsum("ij,kj->ikj", a, b)


@run_debug(MA(2, 3), MB(4, 5))
def test_einsum_outer2d(a, b):
    return einsum("ij,kl->ijkl", a, b)
