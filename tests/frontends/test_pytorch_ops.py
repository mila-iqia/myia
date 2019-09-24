
from copy import copy
from types import FunctionType

import pytest
from pytest import mark

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
from ..multitest import eqtest
from ..test_grad import grad_pipeline, grad_wrap

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

activate_frontend('pytorch')


fwd_compile_pipeline = standard_pipeline


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


def _fwd_test(fn, args, broad_specs, pipeline=standard_pipeline):
    ref_result = fn(*map(copy, args))
    argspec = make_argspec(args, broad_specs)
    res = pipeline.run(input=fn, argspec=argspec)
    myia_fn = res['output']
    myia_result = myia_fn(*map(copy, args))

    assert eqtest(ref_result, myia_result)

    if not isinstance(ref_result, tuple):
        ref_result = (ref_result,)
    if not isinstance(myia_result, tuple):
        myia_result = (myia_result,)
    return [getattr(res, 'shape', ()) for res in myia_result]


def _grad_test(fn, obj, args,
               sens_type,
               broad_specs,
               pipeline=grad_pipeline):

    pytorch_grads = pt_fn_grads(fn, *args)

    sens_type_shape = sens_type
    sens1 = []
    for s in sens_type:
        sens1.append(
            AbstractArray(
                AbstractScalar({TYPE: f32, VALUE: ANYTHING}),
                {SHAPE: tuple(s), TYPE: PyTorchTensor}
            )
        )
    if len(sens1) == 1:
        sens_type = sens1[0]
    else:
        sens_type = AbstractTuple(sens1)

    pipeline = standard_pipeline
    pipeline = pipeline.insert_after('parse', grad_wrap=grad_wrap)
    argspec = make_argspec(args, broad_specs)
    sens_type = to_abstract_test(sens_type)
    if isinstance(obj, FunctionType):
        res = pipeline.run(input=obj, argspec=[*argspec, sens_type])
    else:
        pip = pipeline.configure(parse=False)
        res = pip.run(graph=obj, argspec=[*argspec, sens_type])

    sens = tuple(torch.ones(s) for s in sens_type_shape)
    if len(sens) == 1:
        sens = sens[0]

    myia_grads = res['output'](*args, sens)
    assert eqtest(pytorch_grads, myia_grads, rtol=1e-05, atol=1e-06)


def compare_fwd(*tests, broad_specs=None):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    This uses the full myia pipeline.

    Arguments:
        tests: One or more inputs tuple.

    """
    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)

            _fwd_test(fn, args, broad_specs, pipeline=fwd_compile_pipeline)

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


def compare_bwd(*tests, sens_type=None, pipeline=grad_pipeline):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    Arguments:
        tests: One or more inputs tuple.

    """

    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)

            _grad_test(fn, fn, args, pipeline=pipeline, sens_type=sens_type)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


def compare_fwd_and_bwd(*tests, broad_specs=None,
                        sens_type=None, pipeline=grad_pipeline):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    Arguments:
        tests: One or more inputs tuple.

    """
    def decorate(fn):
        def test(args):
            if not isinstance(args, tuple):
                args = (args,)
            out_shape = _fwd_test(fn, args, broad_specs=broad_specs,
                                  pipeline=fwd_compile_pipeline)
            _grad_test(fn, fn, args, broad_specs=broad_specs,
                       pipeline=pipeline, sens_type=out_shape)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


''' # for when test is not product of "name, args"
def _name_args_helper(name, args):
    return [(name, args) for arg in args]
    #'''


# THIS TEST ALL OPS that are in dir of "torch" or "torch.tensor"
# all_torch_ops = dir(torch)
# all_torch_tensor_ops = dir(torch.Tensor([5.49670]))


torch.manual_seed(123)


all_torch_ops__1_tensor_arg = all_torch_tensor_ops__1_tensor_arg = \
    [
        'exp',
        'log',
        'relu',
        'sigmoid',
        'sum',
        'tanh',
    ]


single_tensor_args = (
    (nn.Parameter(torch.Tensor([2.1]).reshape(()))),
    (nn.Parameter(torch.Tensor([2.1]))),
    (nn.Parameter(torch.Tensor([-2.2]))),
    (nn.Parameter(torch.Tensor(MA(2, 3)))),
)


@pytest.mark.parametrize(
    'name,args',
    [(op, single_tensor_args) for op in all_torch_ops__1_tensor_arg]
)
def test_torch_ops__1_tensor_arg(name, args):
    def fn1(x):
        return getattr(torch, name)(x)

    if not isinstance(args, tuple):
        args = (args,)

    for arg in args:
        out_shape = _fwd_test(fn1, (arg,), broad_specs=None)
        _grad_test(fn1, fn1, (arg,), broad_specs=None, sens_type=out_shape)


@pytest.mark.parametrize(
    'name,args',
    [(op, single_tensor_args) for op in all_torch_tensor_ops__1_tensor_arg]
)
def test_torch_tensor_ops__1_tensor_arg(name, args):
    def fn1(x):
        return getattr(x, name)()

    if not isinstance(args, tuple):
        args = (args,)

    for arg in args:
        out_shape = _fwd_test(fn1, (arg,), broad_specs=None)
        _grad_test(fn1, fn1, (arg,), broad_specs=None, sens_type=out_shape)


all_torch_ops__1_tensor_arg__fwd_only = \
    all_torch_tensor_ops__1_tensor_arg__fwd_only = \
    [
    ]


all_torch_ops__1_tensor_arg__fwd_only.extend([
    # 'zeros_like',  # zl currently only works with pt backend
])


all_torch_tensor_ops__1_tensor_arg__fwd_only.extend([
])

'''
@pytest.mark.parametrize(
    'name,args',
    [(op, single_tensor_args) for op in all_torch_ops__1_tensor_arg__fwd_only]
    )
def test_torch_ops__1_tensor_arg__fwd_only(name, args):
    def fn1(x):
        return getattr(torch, name)(x)

    if not isinstance(args, tuple):
        args = (args,)

    for arg in args:
        _fwd_test(fn1, (arg,))
#'''


'''
@pytest.mark.parametrize(
    'name,args',
    [(op, single_tensor_args) for op
     in all_torch_tensor_ops__1_tensor_arg__fwd_only]
    )
def test_torch_tensor_ops__1_tensor_arg__fwd_only(name, args):
    def fn1(x):
        return getattr(x, name)()

    if not isinstance(args, tuple):
        args = (args,)

    for arg in args:
        _fwd_test(fn1, (arg,))
#'''


all_torch_tensor_ops__1_tensor_arg_1D_and_lower__fwd_only = \
    [
        'item',
    ]


single_tensor_args__1D_and_lower = (
    (nn.Parameter(torch.Tensor([2.1]).reshape(()))),
    (nn.Parameter(torch.Tensor([2.1]))),
)


@pytest.mark.parametrize(
    'name,args',
    [(op, single_tensor_args__1D_and_lower) for op
     in all_torch_tensor_ops__1_tensor_arg_1D_and_lower__fwd_only]
)
def test_torch_item__fwd_only(name, args):
    def fn1(x):
        return getattr(x, name)()

    if not isinstance(args, tuple):
        args = (args,)

    for arg in args:
        _fwd_test(fn1, (arg,), broad_specs=None)


all_torch_ops__2_args = all_torch_tensor_ops____2_args = \
    [
        'reshape',
        'sum',  # version with dim arg to reduce over
        't',
        'view',
    ]


# KEEP THESE IN ALPHABETICAL ORDER ####################################


@compare_fwd((nn.Parameter(torch.Tensor(MA(2, 3)))))
def test_torch_tensor_argmax_1_arg(x):
    return torch.argmax(x)


@compare_fwd((nn.Parameter(torch.Tensor(MA(2, 3))), 1, True),
             (nn.Parameter(torch.Tensor(MA(2, 3))), 0, True),
             broad_specs=(True, False, False))
def test_torch_tensor_argmax_3_arg(x, y, z):
    return torch.argmax(x, y, z)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(3, 4))),
                     torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]])))
def test_torch_gather(x, index):
    return torch.gather(x, 0, index)


@compare_fwd_and_bwd((nn.Parameter(torch.randn(2, 4, 3))))
def test_torch_tensor_get(x):
    return x[:, -3:-1:2, -2]


@compare_fwd_and_bwd((nn.Parameter(torch.randn(2, 4, 3))))
def test_torch_tensor_get2(x):
    return x[1, 2]


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))), 0),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), 1),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), -1),
                     broad_specs=(True, False))
def test_torch_log_softmax(x, y):
    return torch.log_softmax(x, y)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))), 0),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), 1),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), -1),
                     broad_specs=(True, False))
def test_torch_functional_log_softmax(x, y):
    return torch.nn.functional.log_softmax(x, y)


@compare_fwd_and_bwd((nn.Parameter(torch.randn(2, 4, 3))))
def test_torch_tensor_max_1_arg(x):
    return torch.max(x)


@compare_fwd_and_bwd((nn.Parameter(torch.randn(2, 4, 3)), -1, True),
                     (nn.Parameter(torch.randn(2, 4, 3)), 1, True),
                     (nn.Parameter(torch.randn(4)), 0, True),
                     broad_specs=(True, False, False))
def test_torch_tensor_max_3_arg(x, y, z):
    return torch.max(x, y, z)[0]


@compare_fwd_and_bwd((nn.Parameter(torch.randn(2, 4, 3, 5)), False),
                     (nn.Parameter(torch.tensor([[[[1., 2., 3., 4.],
                      [5., 6., 7., 8.], [13., 14., 15., 16.],
                      [9., 10., 11., 12.]]]])),
                      False),
                     broad_specs=(True, False, False, False, False,
                                  False, True))
def test_torch_max_pool2d(x, ri):
    return torch.nn.functional.max_pool2d(x, (2, 2), (1, 1),
                                          0, 1, False, ri)


@compare_fwd((nn.Parameter(torch.randn(2, 4, 3, 5)), True),
             (nn.Parameter(torch.tensor([[[[1., 2., 3., 4.],
              [5., 6., 7., 8.], [13., 14., 15., 16.],
              [9., 10., 11., 12.]]]])),
              True),
             broad_specs=(True, False, False, False, False,
                          False, True))
def test_torch_max_pool2d_return_indices(x, ri):
    return torch.nn.functional.max_pool2d(x, (2, 2), (1, 1),
                                          0, 1, False, ri)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))),
                     torch.tensor([1, 2])))
def test_torch_nll_loss(x, y):
    return torch.nn.functional.nll_loss(x, y)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))),
                      torch.tensor([1, 2]), 'none'),
                     (nn.Parameter(torch.Tensor(MA(2, 3))),
                      torch.tensor([1, 2]), 'sum'),
                     (nn.Parameter(torch.Tensor(MA(2, 3))),
                      torch.tensor([1, 2]), 'mean'),
                     broad_specs=(True, True, False))
def test_torch_nll_loss_reduce_options(x, y, z):
    return torch.nn.functional.nll_loss(x, y, reduction=z)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(torch.randn(2, 3, 4, 5)))))
def test_torch_tensor_permute(x):
    return x.permute((0, 3, 2, 1))


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(1, 3)))))
def test_torch_tensor_pow(x):
    return x ** 2


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))), (-1,)),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), (6,)),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), (2, 3)),
                     (nn.Parameter(torch.Tensor(MA(2, 1))), (2,)),
                     broad_specs=(True, False))
def test_torch_tensor_reshape(x, y):
    return x.reshape(y)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(3, 4))),
                     torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                     nn.Parameter(torch.Tensor(MA(2, 4)))))
def test_torch_scatter(x, index, src):
    return torch.scatter(x, 0, index, src)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(3, 4))),
                     torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                     nn.Parameter(torch.Tensor(MA(2, 4)))))
def test_torch_scatter_add(x, index, src):
    return torch.scatter_add(x, 0, index, src)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(3, 4))),
                     torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
                     nn.Parameter(torch.Tensor([2.1]).reshape(()))))
def test_torch_scatter_broadcast_source(x, index, src):
    return torch.scatter(x, 0, index, src)


# TODO: Need dtype attr for xtype.NDArray to support nonpytorch scalar src
"""
@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(3, 4))),
              torch.tensor([[0, 1, 2, 0], [0, 0, 0, 1]]),
              1.23))
def test_torch_scatter_broadcast_source_nonpytorch_scalar(x, index, src):
    return torch.scatter(x, 0, index, src)
# """


# TODO: NotImplementedError: <_ast.Subscript object at 0x*>
"""
@compare_fwd((nn.Parameter(torch.randn(2, 4, 3)),
             nn.Parameter(torch.Tensor(torch.randn(1)))))
def test_torch_tensor_set(x, y):
    x[0] = y
    return x
# """


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))), 0),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), 1),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), -1),
                     broad_specs=(True, False))
def test_torch_softmax(x, y):
    return torch.softmax(x, y)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(1, 2))), -1),
                     (nn.Parameter(torch.Tensor(MA(2, 1))), 0),
                     (nn.Parameter(torch.Tensor(MA(2, 1))), 1),
                     broad_specs=(True, False))
def test_torch_tensor_squeeze(x, y):
    return x.squeeze(y)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 1)))),
                     broad_specs=(True, False))
def test_torch_tensor_squeeze_all(x):
    return x.squeeze()


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3)))))
def test_torch_sum(x):
    return torch.sum(x)


@compare_fwd((nn.Parameter(torch.Tensor(MA(2, 3)))))
def test_torch_sum_dtype_fwd(x):
    return torch.sum(x, dtype=torch.float64)


""" # TODO: implement bwd for array_cast
@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3)))))
def test_torch_sum_dtype(x):
    return torch.sum(x, dtype=torch.float64)
"""


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3)))))
def test_torch_sum_dim(x):
    return torch.sum(x, -1)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(MA(2, 3))), 1, True),
                     (nn.Parameter(torch.Tensor(MA(2, 3))), 0, False),
                     broad_specs=(True, False, False))
def test_torch_sum_dim_keepdim(x, y, z):
    return torch.sum(x, y, z)


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(torch.randn(2, 3, 4, 5)))))
def test_torch_sum_multi_dim(x):
    return torch.sum(x, (1, 3))


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor(torch.randn(2, 4, 3, 5)))))
def test_torch_tensor_transpose(x):
    return x.transpose(3, 1)


@compare_fwd(())
def test_torch_zeros():
    return torch.zeros(2, 3)


@compare_fwd(())
def test_torch_zeros_dtype():
    return torch.zeros(2, 3, dtype=torch.float64)
