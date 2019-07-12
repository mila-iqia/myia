
import pytest
import numpy as np

from myia import myia, value_and_grad, grad
from myia.debug import traceback  # noqa
#from myia.pipeline import standard_debug_pipeline
#from myia.frontends import load_frontend

from ..common import MA, MB

try:
    import torch
    from torch import nn
except ImportError:
    pytest.skip('no pytorch')


#pt_debug_pipeline = standard_debug_pipeline \
#    .select('parse', 'resolve', 'infer', 'export')


def get_backend_options(args):
    backend = args.backend

    device_type = args.dev

    backend_options_dict = {
        'pytorch': {'device': device_type},
        'nnvm': {'target': device_type, 'device_id': 0},
        'relay': {'target': device_type, 'device_id': 0}
        }

    backend_options = backend_options_dict[backend]

    return backend_options

class Args():

    def __init__(self):
        self.dev = 'cpu'
        # backend used
        self.backend = 'pytorch'
        #self.backend = 'nnvm'
        #self.backend = 'relay'
        # numerical precision
        self.dtype = 'float32'

args = Args()

backend_options = get_backend_options(args)

class Tiny(nn.Module):
    def __init__(self, in_f, out_f):
        super(Tiny, self).__init__()
        #self.W = torch.Tensor(MA(in_f, out_f, dtype=args.dtype))
        self.W = nn.Parameter(torch.Tensor(MA(in_f, out_f, dtype=args.dtype)))

    def forward(self, input):
        return input @ self.W

def test_module_matmul_fwd():

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)

    @myia(backend=args.backend, backend_options=backend_options,
          return_backend=True, frontend='pytorch')
    def step(model, inp):
        return model(inp)
    output = step(model, inp)

    output_expected = torch.Tensor(
        [[ -5.20440006,   0.74380112,  -8.34329891],
         [-18.35869789, -54.33729553,   9.38029861]]
         )

    assert torch.allclose(output, output_expected)


"""
from myia.pipeline import standard_debug_pipeline
from myia.frontends import load_frontend

pt_debug_pipeline = standard_debug_pipeline \
    .select('parse', 'resolve', 'infer', 'export')

def test_module_matmul_fwd__debug():

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)

    pip = pt_debug_pipeline.configure({
        'frontend.name': 'pytorch'
    })
    frontend = load_frontend('pytorch')
    pip = frontend.configure(pip)


    def step(model, inp):
        return model(inp)

    argspec = [
        frontend.to_abstract(model),
        frontend.to_abstract(inp)
    ]

    step_debug = pip.run(input=step, argspec=argspec)['output']

    output = step_debug(model, inp)

    output_expected = torch.Tensor(
        [[ -5.20440006,   0.74380112,  -8.34329891],
         [-18.35869789, -54.33729553,   9.38029861]]
         )

    assert torch.allclose(output, output_expected)
#"""


def test_module_matmul_bwd():

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss.item()

    @myia(backend=args.backend, backend_options=backend_options,
          return_backend=True, frontend='pytorch')
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, dmodel
    loss, grad = step(model, inp, target)

    loss_expected =  3892.92138671875

    assert np.isclose(loss, loss_expected)

    grad_expected = torch.Tensor(
        [[ 116.56797791,  295.31762695,  -22.92411232],
         [ -56.93274689, -349.43133545,  153.96405029],
         [  42.66147232,  202.43022156,  -74.03508759],
         [-346.44219971, -932.08374023,  105.97644043]])

    assert torch.allclose(grad.W, grad_expected)


class MLP_2_Layers(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(MLP_2_Layers, self).__init__()
        self.f1 = nn.Linear(i_size, h_size)
        self.a = nn.Tanh()
        self.f2 = nn.Linear(h_size, o_size)

    def forward(self, x):
        x = self.f1(x)
        x = self.a(x)
        x = self.f2(x)
        x = self.a(x)
        return x


def test_module_2_layer_mlp_fwd():

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers(4, 2, 3)

    @myia(backend=args.backend, backend_options=backend_options,
          return_backend=True, frontend='pytorch')
    def step(model, inp):
        return model(inp)
    output = step(model, inp)

    output_expected = torch.Tensor(
        [[-0.55702960,  0.85518718,  0.13796528],
         [-0.67215765, -0.09247651, -0.38900381]])

    assert torch.allclose(output, output_expected)


def test_module_2_layer_mlp_bwd():

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers(4, 2, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss.item()

    @myia(backend=args.backend, backend_options=backend_options,
          return_backend=True, frontend='pytorch')
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, dmodel
    loss, grad = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_grads = [
        torch.Tensor([[-1.51596880, -7.51286650,  3.24008656,  2.31766868],
        [-5.04396868,  6.33524609, -3.62623000, 16.01710510]]),
        torch.Tensor([1.85057139, 1.95227396]),
        torch.Tensor([[-0.65377355,  4.39202595],
        [-4.45504284,  1.24591899],
        [-1.77709150,  4.90630770]]),
        torch.Tensor([-7.69495678, -6.02438641, -9.53780556])
        ]

    for g, eg in zip(grad.parameters(), expected_grads):
        torch.allclose(g, eg)
