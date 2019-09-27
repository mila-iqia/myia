
from copy import copy
from types import FunctionType

import pytest

from myia.abstract import AbstractArray, from_value
from myia.abstract.data import (
    ANYTHING,
    SHAPE,
    TYPE,
    VALUE,
    AbstractScalar,
    AbstractTuple,
)
from myia.debug.finite_diff import NoTestGrad, clean_args
from myia.frontends import activate_frontend  # noqa: E402
from myia.frontends.pytorch_abstract_types import PyTorchTensor  # noqa: E402
from myia.pipeline import standard_pipeline

from ..common import MA, f32, to_abstract_test
from ..multitest import eqtest, mt, myia_function_test, run, run_no_relay
from ..test_grad import grad_wrap

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

activate_frontend('pytorch')


# Uncomment this line to print values at specific precision
# torch.set_printoptions(precision=8)


@eqtest.register
def eqtest(t1: torch.Tensor, t2, rtol=1e-5, atol=1e-8, **kwargs):
    return torch.allclose(t1, t2, equal_nan=True, atol=atol, rtol=rtol)


@eqtest.register
def eqtest(x1: NoTestGrad, x2, **kwargs):
    return True


def is_tensor_param(x):
    if isinstance(x, torch.Tensor):
        if x.requires_grad:
            return True
    return False


# TODO: should this also return grads with respect to kwargs
def pt_fn_grads(fn, *args, **kwargs):
    output = fn(*args, **kwargs)

    tpa_i = []
    tpa_l = []
    for adx in range(len(args)):
        if is_tensor_param(args[adx]):
            tpa_l.append(args[adx])
            tpa_i.append(adx)

    tensor_param_args = tuple(tpa_l)
    if not isinstance(output, tuple):
        output = (output,)
    grads = list(torch.autograd.grad(
        output, tensor_param_args,
        (torch.ones(o.shape, dtype=o.dtype) for o in output),
        allow_unused=True))

    grad_with_NA = []
    for adx in range(len(args)):
        if adx in tpa_i:
            grad_with_NA.append(grads[0])
            del grads[0]
        else:
            grad_with_NA.append(NoTestGrad(None))

    return tuple(grad_with_NA)


def make_argspec(args, broad_specs):
    if broad_specs is None:
        broad_specs = (True,) * len(args)
    return tuple(from_value(arg, broaden=bs)
                 for bs, arg in zip(broad_specs, clean_args(args)))


def _fwd_and_bwd(fn, args, broad_specs=None, pipeline=standard_pipeline):

    def mksens(s):
        return AbstractArray(
            AbstractScalar({TYPE: f32, VALUE: ANYTHING}),
            {SHAPE: tuple(s), TYPE: PyTorchTensor}
        )

    ref_result = fn(*map(copy, args))
    argspec = make_argspec(args, broad_specs)
    res = pipeline.run(input=fn, argspec=argspec)
    myia_fn = res['output']
    myia_result = myia_fn(*map(copy, args))

    assert eqtest(ref_result, myia_result)

    if isinstance(myia_result, tuple):
        sens_type = AbstractTuple(
            [mksens(res.shape) for res in myia_result]
        )
        sens = tuple(torch.ones(res.shape) for res in myia_result)

    else:
        sens_type = mksens(myia_result.shape)
        sens = torch.ones(myia_result.shape)

    pytorch_grads = pt_fn_grads(fn, *args)

    gpipeline = pipeline.insert_after('parse', grad_wrap=grad_wrap)
    sens_type = to_abstract_test(sens_type)
    assert isinstance(fn, FunctionType)
    res = gpipeline.run(input=fn, argspec=[*argspec, sens_type])

    myia_grads = res['output'](*args, sens)
    assert eqtest(pytorch_grads, myia_grads, rtol=1e-05, atol=1e-06)


@myia_function_test(marks=[pytest.mark.grad], id='grad')
def fwd_and_bwd(self, fn, args, broad_specs=None, pipeline=standard_pipeline):
    _fwd_and_bwd(fn, args, broad_specs, pipeline)


# THIS TEST ALL OPS that are in dir of "torch" or "torch.tensor"
# all_torch_ops = dir(torch)
# all_torch_tensor_ops = dir(torch.Tensor([5.49670]))


torch.manual_seed(123)


single_tensor_arg_tests = (
    fwd_and_bwd(nn.Parameter(torch.Tensor([2.1]).reshape(()))),
    fwd_and_bwd(nn.Parameter(torch.Tensor([2.1]))),
    fwd_and_bwd(nn.Parameter(torch.Tensor([-2.2]))),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3)))),
)


@mt(*single_tensor_arg_tests)
def test_torch_exp(x):
    return torch.exp(x)


@mt(*single_tensor_arg_tests)
def test_torch_log(x):
    return torch.log(x)


@mt(*single_tensor_arg_tests)
def test_torch_relu(x):
    return torch.relu(x)


@mt(*single_tensor_arg_tests)
def test_torch_sigmoid(x):
    return torch.sigmoid(x)


@mt(*single_tensor_arg_tests)
def test_torch_tanh(x):
    return torch.tanh(x)


# KEEP THESE IN ALPHABETICAL ORDER ####################################


@run(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_tensor_argmax_1_arg(x):
    return torch.argmax(x)


@mt(
    run(nn.Parameter(torch.Tensor(MA(2, 3))), 1, True),
    run(nn.Parameter(torch.Tensor(MA(2, 3))), 0, True),
    broad_specs=(True, False, False),
)
def test_torch_tensor_argmax_3_arg(x, y, z):
    return torch.argmax(x, y, z)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 4))),
             torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]))
def test_torch_gather(x, index):
    return torch.gather(x, 0, index)


@mt(
    run(nn.Parameter(torch.Tensor([2.1]).reshape(()))),
    run(nn.Parameter(torch.Tensor([2.1]))),
)
def test_torch_item(x):
    return x.item()


@fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_get(x):
    return x[:, -3:-1:2, -2]


@fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_get2(x):
    return x[1, 2]


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False)
)
def test_torch_log_softmax(x, y):
    return torch.log_softmax(x, y)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False)
)
def test_torch_functional_log_softmax(x, y):
    return torch.nn.functional.log_softmax(x, y)


@fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_max_1_arg(x):
    return torch.max(x)


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3)), -1, True),
    fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3)), 1, True),
    fwd_and_bwd(nn.Parameter(torch.randn(4)), 0, True),
    broad_specs=(True, False, False)
)
def test_torch_tensor_max_3_arg(x, y, z):
    return torch.max(x, y, z)[0]


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(2, 4, 3, 5)), False),
    fwd_and_bwd(nn.Parameter(torch.tensor([[[[1., 2., 3., 4.],
                [5., 6., 7., 8.], [13., 14., 15., 16.],
                [9., 10., 11., 12.]]]])),
                False),
    broad_specs=(True, False, False, False, False, False, True)
)
def test_torch_max_pool2d(x, ri):
    return torch.nn.functional.max_pool2d(x, (2, 2), (1, 1),
                                          0, 1, False, ri)


@mt(
    run_no_relay(nn.Parameter(torch.randn(2, 4, 3, 5)), True),
    run_no_relay(nn.Parameter(torch.tensor([[[[1., 2., 3., 4.],
                                              [5., 6., 7., 8.],
                                              [13., 14., 15., 16.],
                                              [9., 10., 11., 12.]]]])),
                 True),
    broad_specs=(True, False, False, False, False, False, True)
)
def test_torch_max_pool2d_return_indices(x, ri):
    return torch.nn.functional.max_pool2d(x, (2, 2), (1, 1),
                                          0, 1, False, ri)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
             torch.tensor([1, 2]))
def test_torch_nll_loss(x, y):
    return torch.nn.functional.nll_loss(x, y)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
                torch.tensor([1, 2]), 'none'),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
                torch.tensor([1, 2]), 'sum'),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
                torch.tensor([1, 2]), 'mean'),
    broad_specs=(True, True, False)
)
def test_torch_nll_loss_reduce_options(x, y, z):
    return torch.nn.functional.nll_loss(x, y, reduction=z)


@fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(2, 3, 4, 5))))
def test_torch_tensor_permute(x):
    return x.permute((0, 3, 2, 1))


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(1, 3))))
def test_torch_tensor_pow(x):
    return x ** 2


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (-1,)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (6,)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (2, 3)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), (2,)),
    broad_specs=(True, False)
)
def test_torch_tensor_reshape(x, y):
    return x.reshape(y)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 4))),
             torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
             nn.Parameter(torch.Tensor(MA(2, 4))))
def test_torch_scatter(x, index, src):
    return torch.scatter(x, 0, index, src)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 4))),
             torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
             nn.Parameter(torch.Tensor(MA(2, 4))))
def test_torch_scatter_add(x, index, src):
    return torch.scatter_add(x, 0, index, src)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 4))),
             torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
             nn.Parameter(torch.Tensor([2.1]).reshape(())))
def test_torch_scatter_broadcast_source(x, index, src):
    return torch.scatter(x, 0, index, src)


# TODO: Need dtype attr for xtype.NDArray to support nonpytorch scalar src
"""
@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 4))),
              torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
              1.23)
def test_torch_scatter_broadcast_source_nonpytorch_scalar(x, index, src):
    return torch.scatter(x, 0, index, src)
# """


# TODO: NotImplementedError: <_ast.Subscript object at 0x*>
"""
@run(nn.Parameter(torch.randn(2, 4, 3)),
     nn.Parameter(torch.Tensor(torch.randn(1))))
def test_torch_tensor_set(x, y):
    x[0] = y
    return x
# """


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False)
)
def test_torch_softmax(x, y):
    return torch.softmax(x, y)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(1, 2))), -1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), 1),
    broad_specs=(True, False)
)
def test_torch_tensor_squeeze(x, y):
    return x.squeeze(y)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))),
             broad_specs=(True, False))
def test_torch_tensor_squeeze_all(x):
    return x.squeeze()


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum(x):
    return torch.sum(x)


@run(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum_dtype_fwd(x):
    return torch.sum(x, dtype=torch.float64)


""" # TODO: implement bwd for array_cast
@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum_dtype(x):
    return torch.sum(x, dtype=torch.float64)
"""


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum_dim(x):
    return torch.sum(x, -1)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1, True),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0, False),
    broad_specs=(True, False, False)
)
def test_torch_sum_dim_keepdim(x, y, z):
    return torch.sum(x, y, z)


@fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(2, 3, 4, 5))))
def test_torch_sum_multi_dim(x):
    return torch.sum(x, (1, 3))


@fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(2, 4, 3, 5))))
def test_torch_tensor_transpose(x):
    return x.transpose(3, 1)


@run()
def test_torch_zeros():
    return torch.zeros(2, 3)


@run()
def test_torch_zeros_dtype():
    return torch.zeros(2, 3, dtype=torch.float64)
