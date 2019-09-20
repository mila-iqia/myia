
import numpy as np
import pytest

from myia import myia, value_and_grad
from myia.api import to_device
from myia.frontends import activate_frontend
from myia.utils import MyiaTypeError, MyiaValueError

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
        'relay': {'target': device_type, 'device_id': 0}
    }

    backend_options = backend_options_dict[backend]

    return backend_options


@pytest.fixture(params=[
    pytest.param('pytorch'),
    pytest.param('relay'),
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
    .select('resources', 'parse', 'resolve', 'infer', 'export')

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


def test_conv2d_fwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = torch.randn(1, 1, 3, 3, dtype=getattr(torch, args.dtype),
                        requires_grad=True)
    weight = torch.randn(1, 1, 2, 2, dtype=getattr(torch, args.dtype),
                         requires_grad=True)

    def model(inp, w):
        return torch.nn.functional.conv2d(inp, w, None, 1, 0, 1, 1)

    pt_out = model(input, weight)

    @myia(backend=backend, backend_options=backend_options)
    def step(inp, w):
        return model(inp, w)
    my_out = step(input, weight)

    assert torch.allclose(my_out, pt_out)


def test_conv2d_module_grad():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = nn.Parameter(torch.randn(2, 6, 4, 5,
                                     dtype=getattr(torch, args.dtype),
                                     requires_grad=True))
    weight = nn.Parameter(torch.randn(3, 2, 3, 3,
                                      dtype=getattr(torch, args.dtype),
                                      requires_grad=True))

    bias = nn.Parameter(torch.randn(3, dtype=getattr(torch, args.dtype),
                                    requires_grad=True))

    class Conv2dMod(nn.Module):
        def __init__(self):
            super(Conv2dMod, self).__init__()

        def forward(self, inp, w, b):
            return torch.nn.functional.conv2d(
                inp, w, b, (2, 3), (3, 2), (3, 4), 3)

    def cost(_model, inp, w, b):
        value = _model(inp, w, b)
        return torch.sum(value)

    model = Conv2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(_model, inp, w, b):
        _cost, dinp, dw, db = value_and_grad(
            cost, 'inp', 'w', 'b')(_model, inp, w, b)
        return _cost, dinp, dw, db
    my_out_cost, my_out_dinp_grad, my_out_dw_grad, my_out_db_grad = step(
        model, input, weight, bias)

    pt_cost = cost(model, input, weight, bias)
    if input.grad is not None:
        input.grad.data.zero_()
    if weight.grad is not None:
        weight.grad.data.zero_()
    if bias.grad is not None:
        bias.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)
    assert torch.allclose(my_out_dw_grad, weight.grad.data)
    assert torch.allclose(my_out_db_grad, bias.grad.data)


def test_conv2d_module_grad__non_tuple_args():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = nn.Parameter(torch.randn(2, 6, 4, 5,
                                     dtype=getattr(torch, args.dtype),
                                     requires_grad=True))
    weight = nn.Parameter(torch.randn(3, 2, 3, 3,
                                      dtype=getattr(torch, args.dtype),
                                      requires_grad=True))

    bias = nn.Parameter(torch.randn(3, dtype=getattr(torch, args.dtype),
                                    requires_grad=True))

    class Conv2dMod(nn.Module):
        def __init__(self):
            super(Conv2dMod, self).__init__()

        def forward(self, inp, w, b):
            return torch.nn.functional.conv2d(
                inp, w, b, 2, 3, 4, 3)

    def cost(_model, inp, w, b):
        value = _model(inp, w, b)
        return torch.sum(value)

    model = Conv2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(_model, inp, w, b):
        _cost, dinp, dw, db = value_and_grad(
            cost, 'inp', 'w', 'b')(_model, inp, w, b)
        return _cost, dinp, dw, db
    my_out_cost, my_out_dinp_grad, my_out_dw_grad, my_out_db_grad = step(
        model, input, weight, bias)

    pt_cost = cost(model, input, weight, bias)
    if input.grad is not None:
        input.grad.data.zero_()
    if weight.grad is not None:
        weight.grad.data.zero_()
    if bias.grad is not None:
        bias.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)
    assert torch.allclose(my_out_dw_grad, weight.grad.data)
    assert torch.allclose(my_out_db_grad, bias.grad.data)


def test_conv2d_module_grad__group2():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = nn.Parameter(torch.randn(2, 3, 4, 5,
                                     dtype=getattr(torch, args.dtype),
                                     requires_grad=True))
    weight = nn.Parameter(torch.randn(3, 1, 3, 3,
                                      dtype=getattr(torch, args.dtype),
                                      requires_grad=True))

    bias = nn.Parameter(torch.randn(3, dtype=getattr(torch, args.dtype),
                                    requires_grad=True))

    class Conv2dMod(nn.Module):
        def __init__(self):
            super(Conv2dMod, self).__init__()

        def forward(self, inp, w, b):
            return torch.nn.functional.conv2d(
                inp, w, b, (2, 3), (3, 2), (3, 4), 3)

    def cost(_model, inp, w, b):
        value = _model(inp, w, b)
        return torch.sum(value)

    model = Conv2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(_model, inp, w, b):
        _cost, dinp, dw, db = value_and_grad(
            cost, 'inp', 'w', 'b')(_model, inp, w, b)
        return _cost, dinp, dw, db
    my_out_cost, my_out_dinp_grad, my_out_dw_grad, my_out_db_grad = step(
        model, input, weight, bias)

    pt_cost = cost(model, input, weight, bias)
    if input.grad is not None:
        input.grad.data.zero_()
    if weight.grad is not None:
        weight.grad.data.zero_()
    if bias.grad is not None:
        bias.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)
    assert torch.allclose(my_out_dw_grad, weight.grad.data)
    assert torch.allclose(my_out_db_grad, bias.grad.data)


def test_conv2d_module_grad__group3():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = nn.Parameter(torch.randn(2, 1, 4, 5, dtype=getattr(
        torch, args.dtype), requires_grad=True))
    weight = nn.Parameter(torch.randn(
        3, 1, 3, 3, dtype=getattr(torch, args.dtype), requires_grad=True))

    bias = nn.Parameter(torch.randn(3, dtype=getattr(
        torch, args.dtype), requires_grad=True))

    class Conv2dMod(nn.Module):
        def __init__(self):
            super(Conv2dMod, self).__init__()

        def forward(self, inp, w, b):
            return torch.nn.functional.conv2d(
                inp, w, b, (2, 3), (3, 2), (3, 4), 1)

    def cost(_model, inp, w, b):
        value = _model(inp, w, b)
        return torch.sum(value)

    model = Conv2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(_model, inp, w, b):
        _cost, dinp, dw, db = value_and_grad(
            cost, 'inp', 'w', 'b')(_model, inp, w, b)
        return _cost, dinp, dw, db
    my_out_cost, my_out_dinp_grad, my_out_dw_grad, my_out_db_grad = step(
        model, input, weight, bias)

    pt_cost = cost(model, input, weight, bias)
    if input.grad is not None:
        input.grad.data.zero_()
    if weight.grad is not None:
        weight.grad.data.zero_()
    if bias.grad is not None:
        bias.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)
    assert torch.allclose(my_out_dw_grad, weight.grad.data)
    assert torch.allclose(my_out_db_grad, bias.grad.data)


def test_conv2d_module_grad_no_bias():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = nn.Parameter(torch.randn(2, 6, 4, 5, dtype=getattr(
        torch, args.dtype), requires_grad=True))
    weight = nn.Parameter(torch.randn(
        3, 2, 3, 3, dtype=getattr(torch, args.dtype), requires_grad=True))

    class Conv2dMod(nn.Module):
        def __init__(self):
            super(Conv2dMod, self).__init__()

        def forward(self, inp, w):
            return torch.nn.functional.conv2d(
                inp, w, None, (2, 3), (3, 2), (3, 4), 3)

    def cost(_model, inp, w):
        value = _model(inp, w)
        return torch.sum(value)

    model = Conv2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(_model, inp, w):
        _cost, dinp, dw = value_and_grad(cost, 'inp', 'w')(_model, inp, w)
        return _cost, dinp, dw
    my_out_cost, my_out_dinp_grad, my_out_dw_grad = step(model, input, weight)

    pt_cost = cost(model, input, weight)
    if input.grad is not None:
        input.grad.data.zero_()
    if weight.grad is not None:
        weight.grad.data.zero_()
    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)
    assert torch.allclose(my_out_dw_grad, weight.grad.data)


def test_nn_max_pool2d_fwd():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = torch.randn(2, 4, 3, 5, dtype=getattr(torch, args.dtype))

    class MP2dMod(nn.Module):
        def __init__(self):
            super(MP2dMod, self).__init__()
            self.mp1 = nn.MaxPool2d((3, 2), stride=(2, 1))

        def forward(self, inp):
            return self.mp1(inp)

    def cost(model, inp):
        value = model(inp)
        return torch.sum(value)

    model = MP2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(inp):
        return model(inp)
    my_out = step(input)

    pt_out = model(input)

    assert torch.allclose(my_out, pt_out)


def test_nn_max_pool2d_update():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = torch.randn(2, 4, 3, 5, dtype=getattr(torch, args.dtype),
                        requires_grad=True)

    class MP2dMod(nn.Module):
        def __init__(self):
            super(MP2dMod, self).__init__()
            self.mp1 = nn.MaxPool2d((3, 2), stride=(2, 1))

        def forward(self, inp):
            return self.mp1(inp)

    def cost(model, inp):
        value = model(inp)
        return torch.sum(value)

    model = MP2dMod()

    @myia(backend=backend, backend_options=backend_options)
    def step(inp):
        _cost, d_inp = value_and_grad(cost, 'inp')(model, inp)
        return _cost, d_inp
    loss, my_out_dinp_grad = step(input)

    pt_cost = cost(model, input)
    if input.grad is not None:
        input.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)


# TODO: Should this eventually be in a different test file?
#       It's currently here because it needs to have 'torch' imported.
def test_conv_grad_errors():
    from myia.compile.backends.pytorch_conv_grad import conv2d_input

    torch.manual_seed(123)
    input_size = (2, 6, 4, 5)
    weight = torch.randn(3, 2, 3, 3)
    grad_output = torch.ones(2, 3, 2, 1)
    with pytest.raises(ValueError):
        conv2d_input(None, weight, grad_output, (2, 3), (3, 2), (3, 4), 3)
    with pytest.raises(ValueError):
        conv2d_input((2, 2, 6, 4, 5), weight, grad_output,
                     (2, 3), (3, 2), (3, 4), 3)
    with pytest.raises(ValueError):
        conv2d_input(input_size, weight, torch.ones(9, 9, 9, 9),
                     (2, 3), (3, 2), (3, 4), 3)


# TODO: Should this eventually be in a different test file?
#       It's currently here because it needs to have 'torch' imported.
def test_shp_explicit_errors():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    input = torch.ones(2, 3)

    def f1(x):
        return torch.reshape(x, (-2,))

    @myia(backend=backend, backend_options=backend_options)
    def step1(inp):
        return f1(inp)

    with pytest.raises(MyiaValueError):
        ret1 = step1(input)  # noqa: F841

    def f2(x):
        return torch.reshape(x, (-1, -1))

    @myia(backend=backend, backend_options=backend_options)
    def step2(inp):
        return f2(inp)

    with pytest.raises(MyiaValueError):
        ret2 = step2(input)  # noqa: F841

    def f3(x):
        return torch.reshape(x, (2, 2))

    @myia(backend=backend, backend_options=backend_options)
    def step3(inp):
        return f3(inp)

    with pytest.raises(MyiaValueError):
        ret3 = step3(input)  # noqa: F841


# TODO: Should this eventually be in a different test file?
#       It's currently here because it needs to have 'torch' imported.
def test_sum_keepdim_error():
    backend = 'pytorch'
    backend_options = get_backend_options(args, backend)

    input = torch.ones(2, 3)

    def f1(x, kd):
        return torch.sum(x, (1,), kd)

    @myia(backend=backend, backend_options=backend_options)
    def step1(inp, kd):
        return f1(inp, kd)

    with pytest.raises(MyiaTypeError):
        ret1 = step1(input, True)  # noqa: F841


def test_switch_input_types():
    @myia
    def f(x):
        return x * x

    f(torch.ones((2, 2)))
    f(np.ones((2, 2)))
