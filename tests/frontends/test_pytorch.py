
import numpy as np
import pytest
from myia import myia, value_and_grad
from myia.api import to_device
from myia.frontends import activate_frontend
from myia.utils import MyiaTypeError

from ..common import MA

torch = pytest.importorskip("torch")
nn = torch.nn

activate_frontend('pytorch')


def test_pytorch_dtype_to_type():
    from myia.frontends.pytorch import pytorch_dtype_to_type
    with pytest.raises(TypeError):
        pytorch_dtype_to_type("fake_pytorch_type")


# Uncomment this line to print values at specific precision
# torch.set_printoptions(precision=8)


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
    pytest.param('nnvm'),
    # pytest.param('relay'),
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


class Tiny(nn.Module):
    def __init__(self, in_f, out_f):
        super(Tiny, self).__init__()
        self.W = nn.Parameter(torch.Tensor(MA(in_f, out_f, dtype=args.dtype)))
        self.W = nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        return input @ self.W


def test_module_matmul_fwd(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)

    output_expected = torch.Tensor(
        [[-2.92155361,  2.21107101, -4.36360359],
         [6.78069878, -4.06704664,  7.29815578]])

    assert torch.allclose(output, output_expected)


# This will be uncommented once debug VM is compatible with PyTorch
"""
from myia.pipeline import standard_debug_pipeline

pt_debug_pipeline = standard_debug_pipeline \
    .select('parse', 'resolve', 'infer', 'export')

def test_module_matmul_fwd__debug():

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)

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
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, dmodel
    loss, grad = step(model, inp, target)

    loss_expected = 161.0585479736328

    assert np.isclose(loss, loss_expected)

    grad_expected = torch.Tensor(
        [[-15.79940414,   34.22111893,  -16.79670525],
         [82.41101074,  -39.50494003,  100.31848145],
         [-40.12714767,   23.00367165,  -48.50212097],
         [66.75647736, -107.74736023,   74.33836365]])

    assert torch.allclose(grad.W, grad_expected)


def test_module_matmul_update(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Tiny(4, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, model - dmodel

    loss, model = step(model, inp, target)

    loss_expected = 161.0585479736328

    assert np.isclose(loss, loss_expected)

    model_expected = torch.Tensor(
        [[15.42187691, -34.19045258,  16.33688736],
         [-82.06187439,  38.71609116, -99.63981628],
         [39.45422363, -23.73973656,  47.91710663],
         [-66.33718109, 107.40527344, -73.99190521]])

    assert torch.allclose(model.W, model_expected)


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
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers(4, 2, 3)

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)

    output_expected = torch.Tensor(
        [[-0.55702960,  0.85518718,  0.13796528],
         [-0.67215765, -0.09247651, -0.38900381]])

    assert torch.allclose(output, output_expected)


def test_module_2_layer_mlp_bwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

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
        return loss

    @myia(backend=backend, backend_options=backend_options)
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
        assert torch.allclose(g, eg)


def test_module_2_layer_mlp_update():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

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
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, model - dmodel
    loss, model = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_model = [
        torch.Tensor([[1.31208074,  7.52942896, -3.48841572, -2.12911177],
                      [4.61794090, -5.96872425,  3.26280975, -16.41462517]]),
        torch.Tensor([-2.16651487, -1.72582722]),
        torch.Tensor([[0.39250314, -4.12741709],
                      [3.85490060, -1.67493737],
                      [1.51745880, -5.04526806]]),
        torch.Tensor([7.15553093,  6.48739338,  9.37104797])
    ]

    for p, ep in zip(model.parameters(), expected_model):
        assert torch.allclose(p, ep)


def test_module_2_layer_mlp_update__to_device(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers(4, 2, 3)
    target = torch.Tensor([2.5])

    inp = to_device(inp, backend, backend_options)
    model = to_device(model, backend, backend_options)
    target = to_device(target, backend, backend_options)

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, model - dmodel

    loss, model = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_model = [
        torch.Tensor([[1.31208074,  7.52942896, -3.48841572, -2.12911177],
                      [4.61794090, -5.96872425,  3.26280975, -16.41462517]]),
        torch.Tensor([-2.16651487, -1.72582722]),
        torch.Tensor([[0.39250314, -4.12741709],
                      [3.85490060, -1.67493737],
                      [1.51745880, -5.04526806]]),
        torch.Tensor([7.15553093,  6.48739338,  9.37104797])
    ]

    for p, ep in zip(model.parameters(), expected_model):
        assert torch.allclose(p, ep)


def test_pytorch_inference_errors(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    @myia(backend=backend, backend_options=backend_options)
    def step_add(a, b):
        return a + b

    @myia(backend=backend, backend_options=backend_options)
    def step_dot(a, b):
        return a @ b

    _a = torch.Tensor([[1.31208074]])
    _b = np.array([[-8.52928896]], dtype=np.float32)

    with pytest.raises(MyiaTypeError):
        c = step_add(_a, _b)  # noqa: F841

    with pytest.raises(MyiaTypeError):
        c = step_dot(_a, _b)  # noqa: F841


def test_pytorch_scalar(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    _a = 42.759910583496094
    _b = 13.92138671875

    @myia(backend=backend, backend_options=backend_options)
    def step_add(a, b):
        return a + b

    c = step_add(_a, _b)  # noqa: F841


class MLP_2_Layers_Seq(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(MLP_2_Layers_Seq, self).__init__()
        self.f1 = nn.Linear(i_size, h_size)
        self.a = nn.Tanh()
        self.f2 = nn.Linear(h_size, o_size)

        self.f = nn.Sequential(
            self.f1,
            self.a,
            self.f2,
            self.a,
        )

    def forward(self, x):
        x = self.f(x)
        return x


class Linear_Seq(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(Linear_Seq, self).__init__()
        self.f1 = nn.Linear(i_size, h_size)

        self.f = nn.Sequential(
            self.f1,
        )

    def forward(self, x):
        x = self.f(x)
        return x


def test_module_2_layer_mlp_seq_fwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers_Seq(4, 2, 3)

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)

    output_expected = torch.Tensor(
        [[-0.55702960,  0.85518718,  0.13796528],
         [-0.67215765, -0.09247651, -0.38900381]])

    assert torch.allclose(output, output_expected)


"""
def test_module_2_layer_mlp_seq_bwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers_Seq(4, 2, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
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

    print()
    for n, g in grad.named_parameters():
        print("g", n, g)

    for g, eg in zip(grad.parameters(), expected_grads):
        assert torch.allclose(g, eg)


def test_module_linear_seq_bwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    #model = Linear_Seq(4, 2, 3)
    model = MLP_2_Layers_Seq(4, 2, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, dmodel
    loss, grad = step(model, inp, target)

    #assert loss == 42.759910583496094

    expected_grads = [
        torch.Tensor([[-1.51596880, -7.51286650,  3.24008656,  2.31766868],
                      [-5.04396868,  6.33524609, -3.62623000, 16.01710510]]),
        torch.Tensor([1.85057139, 1.95227396]),
        torch.Tensor([[-0.65377355,  4.39202595],
                      [-4.45504284,  1.24591899],
                      [-1.77709150,  4.90630770]]),
        torch.Tensor([-7.69495678, -6.02438641, -9.53780556])
    ]

    # print()
    # for n, g in grad.named_parameters():
    #     print("g", n, g)

    for g, eg in zip(grad.parameters(), expected_grads):
        assert torch.allclose(g, eg)

def test_module_2_layer_mlp_seq_update():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers_Seq(4, 2, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, 'model')(model, inp, target)
        return _cost, model - dmodel
    loss, model = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_model = [
        torch.Tensor([[1.31208074,  7.52942896, -3.48841572, -2.12911177],
                      [4.61794090, -5.96872425,  3.26280975, -16.41462517]]),
        torch.Tensor([-2.16651487, -1.72582722]),
        torch.Tensor([[0.39250314, -4.12741709],
                      [3.85490060, -1.67493737],
                      [1.51745880, -5.04526806]]),
        torch.Tensor([7.15553093,  6.48739338,  9.37104797])
    ]

    for p, ep in zip(model.parameters(), expected_model):
        assert torch.allclose(p, ep)

def test_module_2_layer_mlp_seq_hypermap():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers_Seq(4, 2, 3)
    target = torch.Tensor([2.5])

    def mse(value, target):
        diff = value - target
        return sum(diff * diff)

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    @myia(backend=backend, backend_options=backend_options)
    def step(model):
        return model - model
    model = step(model)
    test_ends_here
    delete_this_test_once_full_sequential_update_is_supported
#"""


"""
# This zeros_like test below is not in test_pytorch_ops because it raises
# "libc++abi.dylib:" "dmlc::Error" because that pipeline can only use nnvm
# TODO: eventually, get torch.zeros_like to work with backends besides pt,
#       and then remove this torch.zeros_like test & use test_pytorch_ops.py


class Z(nn.Module):
    def __init__(self):
        super(Z, self).__init__()

    def forward(self, input):
        return torch.zeros_like(input)


def test_torch_zeros_like():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)
    model = Z()

    inp = torch.Tensor(MA(2, 3, dtype=args.dtype))
    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)
    assert torch.allclose(output, torch.zeros_like(inp))

    inp = torch.Tensor([2.1])
    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)
    assert torch.allclose(output, torch.zeros_like(inp))

    inp = torch.Tensor([2.1]).reshape(())
    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)
    output = step(model, inp)
    assert torch.allclose(output, torch.zeros_like(inp))
#"""
