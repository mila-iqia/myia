
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
from myia.frontends.pytorch_abstract_types import (
    PyTorchTensor,
    pytorch_dtype_to_type,
)
from myia.pipeline import standard_pipeline

from ..common import MA, MB, to_abstract_test
from ..multitest import (
    backend_all,
    backend_no_relay,
    eqtest,
    mt,
    myia_function_test,
    run,
)
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


# sens of 3.21 is used because sens!=1 is more robust test
def _make_sens(o, sens=3.21):
    return torch.ones(o.shape, dtype=o.dtype).fill_(sens)


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
        (_make_sens(o) for o in output),
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


@myia_function_test(marks=[pytest.mark.grad], id='grad')
def _fwd_and_bwd(self, fn, args, broad_specs=None, pipeline=standard_pipeline,
                 backend=False, atol=1e-8, rtol=1e-5,
                 grad_atol=1e-6, grad_rtol=1e-5):
    if backend:
        backend_name = backend[0]
        backend_options = backend[1]

        pipeline = pipeline.configure({
            'resources.backend.name': backend_name,
            'resources.backend.options': backend_options
        })

    def mksens(x):
        return AbstractArray(
            AbstractScalar({TYPE: pytorch_dtype_to_type(x.dtype),
                            VALUE: ANYTHING}),
            {SHAPE: tuple(x.shape), TYPE: PyTorchTensor}
        )

    ref_result = fn(*map(copy, args))
    argspec = make_argspec(args, broad_specs)
    res = pipeline.run(input=fn, argspec=argspec)
    myia_fn = res['output']
    myia_result = myia_fn(*map(copy, args))

    assert eqtest(ref_result, myia_result, atol=atol, rtol=rtol)

    if isinstance(myia_result, tuple):
        sens_type = AbstractTuple(
            [mksens(res) for res in myia_result]
        )
        sens = tuple(_make_sens(res)
                     for res in myia_result)
    else:
        sens_type = mksens(myia_result)
        sens = _make_sens(myia_result)

    pytorch_grads = pt_fn_grads(fn, *args)

    gpipeline = pipeline.insert_after('parse', grad_wrap=grad_wrap)
    sens_type = to_abstract_test(sens_type)
    assert isinstance(fn, FunctionType)
    res = gpipeline.run(input=fn, argspec=[*argspec, sens_type])

    myia_grads = res['output'](*args, sens)
    assert eqtest(pytorch_grads, myia_grads, rtol=grad_rtol, atol=grad_atol)


fwd_and_bwd = _fwd_and_bwd.configure(backend=backend_all)
fwd_and_bwd_no_relay = _fwd_and_bwd.configure(backend=backend_no_relay)


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
def test_torch_abs(x):
    return torch.abs(x)


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
def test_torch_sign(x):
    return torch.sign(x)


@mt(*single_tensor_arg_tests)
def test_torch_sigmoid(x):
    return torch.sigmoid(x)


@mt(*single_tensor_arg_tests, grad_rtol=3e-3)
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


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(1, 6)), 2),
    fwd_and_bwd(nn.Parameter(torch.randn(1, 9)), 2),
    fwd_and_bwd(nn.Parameter(torch.randn(1, 6)), 5),
    fwd_and_bwd(nn.Parameter(torch.randn(1, 6)), 13),
    broad_specs=(True, False)
)
def test_torch_chunk(x, chunks):
    return torch.chunk(x, chunks, dim=1)


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(3, 4, 2)),
                nn.Parameter(torch.randn(3, 5, 2)),
                nn.Parameter(torch.randn(3, 6, 2))),
    broad_specs=(False, False, False)
)
def test_torch_concat(a, b, c):
    return torch.cat((a, b, c), dim=1)


# TODO: uncomment this when bool array compare is merged in pytorch:
"""
http://forum.opennmt.net/t/runtimeerror-subtraction-the-operator-with-a-bool
-tensor-is-not-supported-if-you-are-trying-to-invert-a-mask-use-the-or-bitwise
-not-operator-instead/2994
# """
"""
@run(nn.Parameter(torch.tensor([[0.74, 1., 2.], [0.3, 0.0, 0.6]])),
     nn.Parameter(torch.tensor([[0.74, 1., 2.], [0.7, 0.0, -0.6]])))
def test_torch_eq(x, y):
    return torch.eq(x, y)
# """


@fwd_and_bwd(torch.randn(1, 1, 3, 3, dtype=torch.float32, requires_grad=True),
             torch.randn(1, 1, 2, 2, dtype=torch.float32, requires_grad=True))
def test_conv2d_no_dil(inp, w):
    return torch.nn.functional.conv2d(inp, w, None, 1, 0, 1, 1)


@pytest.mark.xfail
@fwd_and_bwd(torch.randn(1, 1, 3, 3, dtype=torch.float32, requires_grad=True),
             torch.randn(1, 1, 2, 2, dtype=torch.float32, requires_grad=True))
def test_conv2d_no_dil_stride(inp, w):
    return torch.nn.functional.conv2d(inp, w, None, (2, 3))


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(2, 6, 4, 5, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, 2, 3, 3, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, dtype=torch.float32))),
    fwd_and_bwd(nn.Parameter(torch.randn(2, 3, 4, 5, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, 1, 3, 3, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, dtype=torch.float32))),
    fwd_and_bwd(nn.Parameter(torch.randn(2, 6, 4, 5, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, 2, 3, 3, dtype=torch.float32)),
                None),
    backend=backend_no_relay
)
def test_torch_conv2d(inp, w, b):
    value = torch.nn.functional.conv2d(inp, w, b, (2, 3), (3, 2), (3, 4), 3)
    return torch.sum(value)


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(2, 6, 4, 5, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, 2, 3, 3, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, dtype=torch.float32))),
    fwd_and_bwd(nn.Parameter(torch.randn(2, 6, 4, 5, dtype=torch.float32)),
                nn.Parameter(torch.randn(3, 2, 3, 3, dtype=torch.float32)),
                None),
    backend=backend_no_relay
)
def test_torch_conv2d__non_tuple_args(inp, w, b):
    value = torch.nn.functional.conv2d(inp, w, b, 2, 3, 4, 3)
    return torch.sum(value)


@fwd_and_bwd_no_relay(
    nn.Parameter(torch.randn(2, 1, 4, 5, dtype=torch.float32)),
    nn.Parameter(torch.randn(3, 1, 3, 3, dtype=torch.float32)),
    nn.Parameter(torch.randn(3, dtype=torch.float32)))
def test_torch_conv2d__group3(inp, w, b):
    value = torch.nn.functional.conv2d(inp, w, b, (2, 3), (3, 2), (3, 4), 1)
    return torch.sum(value)


@mt(
    fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(torch.randn(3, 5))),
                         nn.Parameter(torch.randint(5, (3,)),
                                      requires_grad=False),
                         'mean'),
    fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(torch.randn(3, 5))),
                         nn.Parameter(torch.randint(5, (3,)),
                                      requires_grad=False),
                         'none'),
    fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(torch.randn(3, 5))),
                         nn.Parameter(torch.randint(5, (3,)),
                                      requires_grad=False),
                         'sum'),
    broad_specs=(True, True, False)
)
def test_torch_cross_entropy(inp, target, reduction):
    return F.cross_entropy(inp, target, reduction=reduction)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(7, 3)))),
    fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(5, 8)))),
    fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(10)))),
)
def test_torch_detach(x):
    # Example copied from here (2019/12/03):
    # http://www.bnikolic.co.uk/blog/pytorch-detach.html
    y = x ** 2
    z = x.detach() ** 3
    r = (y + z).sum()
    return r


@fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(MA(3, 4))),
                      torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]))
def test_torch_gather(x, index):
    return torch.gather(x, 0, index)


@mt(
    run(nn.Parameter(torch.Tensor([2.1]).reshape(()))),
    run(nn.Parameter(torch.Tensor([2.1]))),
)
def test_torch_item(x):
    return x.item()


@fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_get(x):
    return x[:, -3:-1:2, -2]


@fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_get2(x):
    return x[1, 2]


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0, rtol=1e-4),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False),
    backend=backend_no_relay
)
def test_torch_log_softmax(x, y):
    return torch.log_softmax(x, y)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0, rtol=1e-4),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False),
    backend=backend_no_relay
)
def test_torch_functional_log_softmax(x, y):
    return torch.nn.functional.log_softmax(x, y)


@fwd_and_bwd(torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             torch.randn(4, 4, dtype=torch.float32, requires_grad=True),
             grad_atol=1e-5
             )
def test_lstm_cell(inp, hx, cx, w_ih, w_hh, b_ih, b_hh):
    return torch.lstm_cell(inp, (hx, cx), w_ih, w_hh, b_ih, b_hh)


@fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 4, 3)))
def test_torch_tensor_max_1_arg(x):
    return torch.max(x)


@mt(
    fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 4, 3)), -1, True),
    fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 4, 3)), 1, True),
    fwd_and_bwd_no_relay(nn.Parameter(torch.randn(4)), 0, True),
    run(nn.Parameter(torch.randn(2, 4, 3)), -1, True),
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


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_mean(x):
    return torch.mean(x)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
             nn.Parameter(torch.Tensor(MB(2, 3))))
def test_torch_mse_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)


@fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(MA(2, 3))),
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
    broad_specs=(True, True, False),
    backend=backend_no_relay
)
def test_torch_nll_loss_reduce_options(x, y, z):
    return torch.nn.functional.nll_loss(x, y, reduction=z)


@fwd_and_bwd_no_relay(nn.Parameter(torch.randn(2, 3, dtype=torch.float64)),
                      torch.tensor([1, 2]))
def test_torch_nll_loss_reduce_cast(x, y):
    return torch.nn.functional.nll_loss(x, y, reduction='mean')


@fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(2, 3, 4, 5))))
def test_torch_tensor_permute(x):
    return x.permute((0, 3, 2, 1))


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(1, 2))))
def test_torch_tensor_pow(x):
    return x ** 2


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (-1,)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (6,)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (2, 3)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), (-1, 6)),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), (2,)),
    broad_specs=(True, False)
)
def test_torch_tensor_reshape(x, y):
    return x.reshape(y)


@fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(MA(3, 4))),
                      torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                      nn.Parameter(torch.Tensor(MA(2, 4))))
def test_torch_scatter(x, index, src):
    return torch.scatter(x, 0, index, src)


@fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(MA(3, 4))),
                      torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                      nn.Parameter(torch.Tensor(MA(2, 4))))
def test_torch_scatter_add(x, index, src):
    return torch.scatter_add(x, 0, index, src)


@fwd_and_bwd_no_relay(nn.Parameter(torch.Tensor(MA(3, 4))),
                      torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                      nn.Parameter(torch.Tensor([2.1]).reshape(())))
def test_torch_scatter_broadcast_source(x, index, src):
    return torch.scatter(x, 0, index, src)


@fwd_and_bwd_no_relay(nn.Parameter(torch.randn(3, 4, dtype=torch.float64)),
                      torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                      1.23)
def test_torch_scatter_broadcast_source_nonpytorch_scalar(x, index, src):
    return torch.scatter(x, 0, index, src)


# TODO: NotImplementedError: <_ast.Subscript object at 0x*>
"""
@run(nn.Parameter(torch.randn(2, 4, 3)),
     nn.Parameter(torch.Tensor(torch.randn(1))))
def test_torch_tensor_set(x, y):
    x[0] = y
    return x
# """


@run(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_size(x):
    return x.size(-1), x.size()


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))),
             nn.Parameter(torch.Tensor(MB(2, 3))))
def test_torch_smooth_l1_loss(x, y):
    return torch.nn.functional.smooth_l1_loss(x, y, reduction='mean')


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))), -1),
    broad_specs=(True, False),
    backend=backend_no_relay
)
def test_torch_softmax(x, y):
    return torch.softmax(x, y)


@fwd_and_bwd(nn.Parameter(torch.randn(3, 9, 2)))
def test_torch_split(x):
    return torch.split(x, (4, 3, 2), dim=1)


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(3, 2))), -1),
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


@mt(
    fwd_and_bwd(nn.Parameter(torch.randn(5, 4, 2)),
                nn.Parameter(torch.randn(5, 4, 2)),
                nn.Parameter(torch.randn(5, 4, 2))),
    broad_specs=(False, False, False)
)
def test_torch_stack(a, b, c):
    return torch.stack((a, b, c), dim=1)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_std(x):
    return torch.std(x, dim=-1)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum(x):
    return torch.sum(x)


# TODO: need pytorch-cpu=1.2.0 or higher to install to run this test
"""
@run(torch.BoolTensor([[True, False, False], [False, False, True]]))
def test_torch_sum_bool(x):
    return torch.sum(x)
# """


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_sum_dtype(x):
    return torch.sum(x, dtype=torch.float64)


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


@mt(
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(1, 2))), -1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), 0),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), 1),
    fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 1))), 2),
    broad_specs=(True, False)
)
def test_torch_tensor_unsqueeze(x, y):
    return x.unsqueeze(y)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_var(x):
    return torch.var(x)


@fwd_and_bwd(nn.Parameter(torch.Tensor(MA(2, 3))))
def test_torch_var_dim(x):
    return torch.var(x, dim=-1)


@fwd_and_bwd(nn.Parameter(torch.Tensor(torch.randn(2, 4, 3))),
             nn.Parameter(torch.Tensor(torch.randn(4, 3, 2))))
def test_torch_tensor_view_as(x, y):
    return x.view_as(y)


@run()
def test_torch_zeros():
    return torch.zeros(2, 3)


@run()
def test_torch_zeros_dtype():
    return torch.zeros(2, 3, dtype=torch.float64)
