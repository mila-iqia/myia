
import pytest
import numpy as np

from myia import myia, value_and_grad, grad
from myia.debug import traceback  # noqa
from myia.abstract import MyiaTypeError
from myia.api import to_device

from ..common import MA

torch = pytest.importorskip("torch")
nn = torch.nn

from myia.frontends.pytorch import pytorch_dtype_to_type  # noqa: E402

from myia.frontends import activate_frontend  # noqa: E402
activate_frontend('pytorch')


def test_pytorch_dtype_to_type():
    with pytest.raises(TypeError):
        pytorch_dtype_to_type("fake_pytorch_type")


# Uncomment this line to print values at specific precision
torch.set_printoptions(precision=10)


def get_backend_options(args, backend):
    device_type = args.dev

    backend_options_dict = {
        'pytorch': {'device': device_type},
        'nnvm': {'target': device_type, 'device_id': 0},
        'relay': {'target': device_type, 'device_id': 0}
    }

    backend_options = backend_options_dict[backend]

    return backend_options


# TODO: add relay support
# TODO: maybe fixture for return_backend=True and return_backend=False
@pytest.fixture(params=[
    pytest.param('pytorch'),
    pytest.param('nnvm')
])
def _backend_fixture(request):
    return request.param


class Args():

    def __init__(self):
        # device used
        self.dev = 'cpu'
        # backend used
        self.backend = 'pytorch'
        # numerical precision
        self.dtype = 'float32'


args = Args()

from pytest import mark
from copy import copy
from myia.abstract import from_value
from myia.pipeline import standard_pipeline
compile_pipeline = standard_pipeline

#def parse_compare_exp(*tests, optimize=True, python=True, profile=no_prof):
def compare_fwd(*tests, optimize=True, python=True, profile=False):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    This uses the full myia pipeline.

    Arguments:
        tests: One or more inputs tuple.

    """
    fwd_pipeline = compile_pipeline if optimize else \
        compile_pipeline.configure({'opt.phases.main': []})

    def decorate(fn):
        def test(args):
            nonlocal profile
            if not isinstance(args, tuple):
                args = (args,)

            _fwd_test(fn, args, pipeline=fwd_pipeline, optimize=optimize, python=python, profile=profile)

        m = mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


import pytest
from pytest import mark
import numpy as np
from types import FunctionType
from dataclasses import dataclass

from myia.abstract import from_value, AbstractJTagged, InferenceError
from myia.api import myia
from myia.pipeline import standard_resources, standard_pipeline
from myia.composite import grad, value_and_grad
from myia.debug.finite_diff import GradTester, NoTestGrad, clean_args
from myia.grad import J as realJ
from myia.pipeline import pipeline_function, PipelineDefinition, steps
from myia.pipeline.steps import Validator
from myia.prim import ops as P, Primitive
from myia.prim.py_implementations import J, scalar_add, scalar_mul, \
    array_to_scalar, scalar_to_array, array_map, array_reduce, scalar_div, \
    distribute, dot, reshape, transpose, scalar_cast
from myia.prim.py_implementations import py_registry as pyi
from myia.validate import whitelist, validate_abstract
from myia.ir import Graph

from ..common import f64, f32, u64, MA, MB, to_abstract_test, AA


grad_whitelist = whitelist | {P.J, P.Jinv}


@validate_abstract.variant
def grad_validate_abstract(self, t: AbstractJTagged):
    pass


step_grad_validate = Validator.partial(
    whitelist=grad_whitelist,
    validate_abstract=grad_validate_abstract
)


@pipeline_function
def grad_wrap(self, graph):
    if isinstance(graph, Primitive):
        jg = realJ(graph, self.resources)
        g = grad.make_gf(jg, jg.parameters, wrt=range(len(jg.parameters)),
                         dbg=jg.debug, sens_param=True)
    else:
        g = grad.make_gf(graph, graph.parameters,
                         wrt=range(len(graph.parameters)),
                         dbg=graph.debug, sens_param=True,
                         apply_j=True)
    return g


grad_pipeline = PipelineDefinition(
    resources=standard_resources,
    steps=dict(
        parse=steps.step_parse,
        resolve=steps.step_resolve,
        infer=steps.step_infer,
        specialize=steps.step_specialize,
        opt=steps.step_debug_opt,
        validate=step_grad_validate,
        export=steps.step_debug_export,
    )
)

def pt_fn_wrap(fn):
    def pt_fn(*args, **kwargs):
        a = fn(*args, **kwargs)
        #print("args", args)
        try:
            args[0].grad.data.zero_()
        except:
            pass
        a.backward()
        return args
    return pt_fn

from myia.abstract.data import SHAPE, TYPE, VALUE, ANYTHING, AbstractScalar
from myia.frontends.pytorch import AbstractPyTorchTensor

APT_loss = AbstractPyTorchTensor(AbstractScalar({TYPE: f32, VALUE: ANYTHING}), {SHAPE: (1,)})


def _fwd_test(fn, args, pipeline=standard_pipeline, optimize=True, python=True, profile=False):
    if python:
        ref_result = fn(*map(copy, args))
    argspec = tuple(from_value(arg, broaden=True) for arg in args)
    if profile is True:
        profile = Profile()
    #res = pipeline.run(input=fn, argspec=argspec, profile=profile)
    res = pipeline.run(input=fn, argspec=argspec)
    #profile.print()
    myia_fn = res['output']
    myia_result = myia_fn(*map(copy, args))
    if python:
        print("ref_result", ref_result)
        print("myia_result", myia_result)
        assert torch.allclose(ref_result, myia_result)


def _grad_test(fn, obj, args,
               #sens_type=f32,
               sens_type=APT_loss,
               pipeline=grad_pipeline,
               rel_error=1e-3):

    pt_grad = pt_fn_wrap(fn)
    pytorch_grad = pt_grad(*args)[0].grad

    pipeline = standard_pipeline
    pipeline = pipeline.insert_after('parse', grad_wrap=grad_wrap)
    #pipeline = standard_pipeline
    argspec = tuple(from_value(arg, broaden=True) for arg in clean_args(args))
    sens_type = to_abstract_test(sens_type)
    if isinstance(obj, FunctionType):
        res = pipeline.run(input=obj, argspec=[*argspec, sens_type])
        #res = pipeline.run(input=obj, argspec=[*argspec])
    else:
        pip = pipeline.configure(parse=False)
        res = pip.run(graph=obj, argspec=[*argspec, sens_type])

    if sens_type == APT_loss:
        sens = torch.Tensor([1.0])
    elif sens_type == AbstractScalar({TYPE: f32, VALUE: ANYTHING}):
        sens = 1.0

    myia_grad = res['output'](*args, sens)

    print("pytorch_grad", pytorch_grad)
    print("myia_grad", myia_grad)

    assert torch.allclose(pytorch_grad, myia_grad[0])

def compare_bwd(*tests, sens_type=APT_loss, pipeline=grad_pipeline, rel_error=1e-3):
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

            _grad_test(fn, fn, args, pipeline=pipeline, rel_error=rel_error, sens_type=sens_type)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate


def compare_fwd_and_bwd(*tests, optimize=True, python=True, profile=False, sens_type=APT_loss, pipeline=grad_pipeline, rel_error=1e-3):
    """Decorate a function to parse and run it against pure Python.

    Returns a unit test that will parse the function, and then for
    each `inputs` tuple in `tests` it will check that the pure Python,
    undecorated function returns that same output.

    Arguments:
        tests: One or more inputs tuple.

    """

    fwd_pipeline = compile_pipeline if optimize else \
        compile_pipeline.configure({'opt.phases.main': []})

    def decorate(fn):
        def test(args):
            #TODO: "nonlocal profile" maybe should be be made nonlocal inside of _fwd_test
            #nonlocal profile
            if not isinstance(args, tuple):
                args = (args,)
            _fwd_test(fn, args, pipeline=fwd_pipeline, optimize=optimize, python=python, profile=profile)
            _grad_test(fn, fn, args, pipeline=pipeline, rel_error=rel_error, sens_type=sens_type)

        m = pytest.mark.parametrize('args', list(tests))(test)
        m.__orig__ = fn
        return m
    return decorate
    

'''
def _name_args_helper(name, args):
    return [('sigmoid', args) for arg in args]

#TODO
@pytest.mark.parametrize('name,arg', 
                    _name_args_helper('sigmoid', ((nn.Parameter(torch.Tensor([2.1]))), 
                                                (nn.Parameter(torch.Tensor([-2.0]))))
                                     )
                    )
#@pytest.mark.parametrize()
def test_ops(name, arg):
    code = f"""
    def fn1(x):
        return x.{name}()
    def fn2(x):
        return torch.{name}(x)
    """

    ns = {}
    exec(code, ns)
    fn1 = ns['fn1']

    _fwd_test(fn1, args, pipeline=fwd_pipeline, optimize=optimize, python=python, profile=profile)
    _grad_test(fn1, fn1, args, pipeline=pipeline, rel_error=rel_error, sens_type=sens_type)

    fn2 = ns['fn2']
    
    _fwd_test(fn2, args, pipeline=fwd_pipeline, optimize=optimize, python=python, profile=profile)
    _grad_test(fn2, fn2, args, pipeline=pipeline, rel_error=rel_error, sens_type=sens_type)
#'''

"""
# THIS TEST ALL OPS that are in dir of "torch" or "torch.tensor"

all_torch_ops = dir(torch)
all_torch_tensor_ops = dir(torch.Tensor([5.49670]))
torch_tensor_args_exp = ((nn.Parameter(torch.Tensor([2.1]))))

@pytest.mark.parametrize(
    'name,args',
    [(op, torch_tensor_args_exp) for op in all_torch_ops]
    )
#@pytest.mark.timeout(1)
def test_torch_ops(name, args):
    def fn1(x):
        return getattr(torch, name)(x)

    if not isinstance(args, tuple):
        args = (args,)

    _fwd_test(fn1, args)
    _grad_test(fn1, fn1, args)


@pytest.mark.parametrize(
    'name,args',
    [(op, torch_tensor_args_exp) for op in all_torch_tensor_ops]
    )
def test_torch_tensor_ops(name, args):
    def fn1(x):
        return getattr(x, name)()

    if not isinstance(args, tuple):
        args = (args,)

    _fwd_test(fn1, args)
    _grad_test(fn1, fn1, args)
#"""


@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_torch_tanh(x):
    return torch.tanh(x)

@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_tensor_tanh(x):
    return x.tanh()

@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_torch_sigmoid(x):
    return torch.sigmoid(x)

@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_tensor_sigmoid(x):
    return x.sigmoid()

@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_torch_exp(x):
    return torch.exp(x)

@compare_fwd_and_bwd((nn.Parameter(torch.Tensor([2.1]))), 
                     (nn.Parameter(torch.Tensor([-2.0]))))
def test_op_tensor_exp(x):
    return x.exp()
