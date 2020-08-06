import numpy as np
import pytest

from myia import grad, myia, value_and_grad
from myia.api import to_device
from myia.frontends import activate_frontend
from myia.frontends.pytorch import tensor_pytorch_aliasable
from myia.utils import MyiaInputTypeError, MyiaTypeError, MyiaValueError

from ..common import MA
from ..multitest import eqtest, run

torch = pytest.importorskip("torch")
nn = torch.nn

activate_frontend("pytorch")


@eqtest.register
def eqtest(t1: torch.Tensor, t2, rtol=1e-5, atol=1e-8, **kwargs):
    return torch.allclose(t1, t2, equal_nan=True, atol=atol, rtol=rtol)


def test_pytorch_dtype_to_type():
    from myia.frontends.pytorch import pytorch_dtype_to_type

    with pytest.raises(TypeError):
        pytorch_dtype_to_type("fake_pytorch_type")


# Uncomment this line to print values at specific precision
# torch.set_printoptions(precision=8)


def get_backend_options(args, backend):
    device_type = args.dev

    backend_options_dict = {
        "pytorch": {"device": device_type},
        "relay": {"target": device_type, "device_id": 0},
    }

    backend_options = backend_options_dict[backend]

    return backend_options


@pytest.fixture(params=[pytest.param("pytorch"), pytest.param("relay")])
def _backend_fixture(request):
    return request.param


class Args:
    def __init__(self):
        # device used
        self.dev = "cpu"
        # backend used (set 'relay' by default)
        self.backend = "relay"
        # numerical precision
        self.dtype = "float32"


args = Args()


class Tiny(nn.Module):
    def __init__(self, in_f, out_f):
        super(Tiny, self).__init__()
        self.W = nn.Parameter(torch.Tensor(MA(in_f, out_f, dtype=args.dtype)))
        self.W = nn.init.xavier_uniform_(self.W)

    def forward(self, input):
        return input @ self.W


@run(Tiny(4, 3), torch.tensor(MA(2, 4, dtype="float32")))
def test_module_matmul_fwd(model, inp):
    return model(inp)


def test_module_matmul_bwd(_backend_fixture):
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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, dmodel

    loss, grad = step(model, inp, target)

    loss_expected = 161.0585479736328

    assert np.isclose(loss, loss_expected)

    grad_expected = torch.Tensor(
        [
            [-15.79940414, 34.22111893, -16.79670525],
            [82.41101074, -39.50494003, 100.31848145],
            [-40.12714767, 23.00367165, -48.50212097],
            [66.75647736, -107.74736023, 74.33836365],
        ]
    )

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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, model - dmodel

    loss, model = step(model, inp, target)

    loss_expected = 161.0585479736328

    assert np.isclose(loss, loss_expected)

    model_expected = torch.Tensor(
        [
            [15.42187691, -34.19045258, 16.33688736],
            [-82.06187439, 38.71609116, -99.63981628],
            [39.45422363, -23.73973656, 47.91710663],
            [-66.33718109, 107.40527344, -73.99190521],
        ]
    )

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


def test_module_2_layer_mlp_fwd(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers(4, 2, 3)

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)

    output = step(model, inp)

    output_expected = torch.Tensor(
        [
            [-0.55702960, 0.85518718, 0.13796528],
            [-0.67215765, -0.09247651, -0.38900381],
        ]
    )

    assert torch.allclose(output, output_expected)


def test_module_2_layer_mlp_bwd(_backend_fixture):
    backend = _backend_fixture
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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, dmodel

    loss, grad = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_grads = [
        torch.Tensor(
            [
                [-1.51596880, -7.51286650, 3.24008656, 2.31766868],
                [-5.04396868, 6.33524609, -3.62623000, 16.01710510],
            ]
        ),
        torch.Tensor([1.85057139, 1.95227396]),
        torch.Tensor(
            [
                [-0.65377355, 4.39202595],
                [-4.45504284, 1.24591899],
                [-1.77709150, 4.90630770],
            ]
        ),
        torch.Tensor([-7.69495678, -6.02438641, -9.53780556]),
    ]

    for g, eg in zip(grad.parameters(), expected_grads):
        assert torch.allclose(g, eg)


def test_module_2_layer_mlp_update(_backend_fixture):
    backend = _backend_fixture
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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, model - dmodel

    loss, model = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_model = [
        torch.Tensor(
            [
                [1.31208074, 7.52942896, -3.48841572, -2.12911177],
                [4.61794090, -5.96872425, 3.26280975, -16.41462517],
            ]
        ),
        torch.Tensor([-2.16651487, -1.72582722]),
        torch.Tensor(
            [
                [0.39250314, -4.12741709],
                [3.85490060, -1.67493737],
                [1.51745880, -5.04526806],
            ]
        ),
        torch.Tensor([7.15553093, 6.48739338, 9.37104797]),
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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, model - dmodel

    loss, model = step(model, inp, target)

    assert loss == 42.759910583496094

    expected_model = [
        torch.Tensor(
            [
                [1.31208074, 7.52942896, -3.48841572, -2.12911177],
                [4.61794090, -5.96872425, 3.26280975, -16.41462517],
            ]
        ),
        torch.Tensor([-2.16651487, -1.72582722]),
        torch.Tensor(
            [
                [0.39250314, -4.12741709],
                [3.85490060, -1.67493737],
                [1.51745880, -5.04526806],
            ]
        ),
        torch.Tensor([7.15553093, 6.48739338, 9.37104797]),
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

        self.f = nn.Sequential(self.f1, self.a, self.f2, self.a)

    def forward(self, x):
        x = self.f(x)
        return x


class Linear_Seq(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(Linear_Seq, self).__init__()
        self.f1 = nn.Linear(i_size, h_size)

        self.f = nn.Sequential(self.f1)

    def forward(self, x):
        x = self.f(x)
        return x


@pytest.mark.xfail
def test_module_2_layer_mlp_seq_fwd(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = MLP_2_Layers_Seq(4, 2, 3)

    @myia(backend=backend, backend_options=backend_options)
    def step(model, inp):
        return model(inp)

    output = step(model, inp)

    output_expected = torch.Tensor(
        [
            [-0.55702960, 0.85518718, 0.13796528],
            [-0.67215765, -0.09247651, -0.38900381],
        ]
    )

    assert torch.allclose(output, output_expected)


@pytest.mark.xfail
def test_module_linear_seq_bwd(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    inp = torch.Tensor(MA(2, 4, dtype=args.dtype))
    model = Linear_Seq(4, 2, 3)
    target = torch.Tensor([2.5])

    # Leave this here, it will be needed again for when we investigate how
    # to automatically remove duplicates from grads of sequential
    """
    from myia.abstract.aliasing import find_aliases
    al = find_aliases(model, aliasable=tensor_pytorch_aliasable)
    print("alias", al)
    # """

    def mse(value, target):
        diff = value - target
        return (diff * diff).sum()

    def cost(model, inp, target):
        value = model(inp)
        loss = mse(value, target)
        return loss

    pt_cost = cost(model, inp, target)

    @myia(
        backend=backend,
        backend_options=backend_options,
        alias_tracker=tensor_pytorch_aliasable,
    )
    def step(model, inp, target):
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, dmodel

    loss, grad = step(model, inp, target)

    pt_cost = cost(model, inp, target)
    if model.f1.weight.grad is not None:
        model.f1.weight.grad.data.zero_()
    if model.f1.bias.grad is not None:
        model.f1.bias.grad.data.zero_()
    pt_cost.backward()

    for n, p in model.named_parameters():
        m_p = grad
        for a in tuple(n.split(".")):
            m_p = getattr(m_p, a)
        assert torch.allclose(p.grad.data, m_p)


class Linear_List(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(Linear_List, self).__init__()
        self.f1 = nn.Linear(i_size, h_size)

        self.f = [self.f1]

    def forward(self, x):
        x = self.f[0](x)
        return x


def test_alias_list_error(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    def g(xs, y):
        res = 0
        for x in xs:
            res = res + x
        return sum(res)

    @myia(
        backend=backend,
        backend_options=backend_options,
        alias_tracker=tensor_pytorch_aliasable,
    )
    def f(xs, y):
        return grad(g)(xs, y)

    o = torch.ones((1, 3))

    a = o * 3
    b = o * 4
    c = o * 5
    e = o * 7

    with pytest.raises(MyiaInputTypeError):
        print(f([a, b, c, a], e))


def test_nn_max_pool2d_fwd(_backend_fixture):
    backend = _backend_fixture
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


def test_nn_max_pool2d_update(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    torch.manual_seed(123)

    input = torch.randn(
        2, 4, 3, 5, dtype=getattr(torch, args.dtype), requires_grad=True
    )

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
        _cost, d_inp = value_and_grad(cost, "inp")(model, inp)
        return _cost, d_inp

    loss, my_out_dinp_grad = step(input)

    pt_cost = cost(model, input)
    if input.grad is not None:
        input.grad.data.zero_()

    pt_cost.backward()

    assert torch.allclose(my_out_dinp_grad, input.grad.data)


# TODO: Should this eventually be in a different test file?
#       It's currently here because it needs to have 'torch' imported.
def test_shp_explicit_errors(_backend_fixture):
    backend = _backend_fixture
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
def test_sum_keepdim_error(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    input = torch.ones(2, 3)

    def f1(x, kd):
        return torch.sum(x, (1,), kd)

    @myia(backend=backend, backend_options=backend_options)
    def step1(inp, kd):
        return f1(inp, kd)

    with pytest.raises(MyiaTypeError):
        ret1 = step1(input, True)  # noqa: F841


def test_switch_input_types(_backend_fixture):
    backend = _backend_fixture
    backend_options = get_backend_options(args, backend)

    @myia(backend=backend, backend_options=backend_options)
    def f(x):
        return x * x

    f(torch.ones((2, 2)))
    f(np.ones((2, 2)))


# This is mostly here to cover inst_tuple_setitem method in myia.compile.vm
def test_optim_setitem(_backend_fixture):
    from myia.abstract import macro
    from myia.operations import primitives as P
    from myia.ir import sexp_to_node
    from myia.lib import setter_from_getter

    def update_sgd(p, g):
        return p - 0.01 * g

    @macro
    async def update(info, model_ref, dmodel_ref, update_rule_ref):
        new_model = model_ref.node
        dmodel = dmodel_ref.node
        update_rule = update_rule_ref.node

        p = new_model
        g = dmodel

        p = (P.record_getitem, p, "W")
        g = (P.record_getitem, g, "W")

        p_node = sexp_to_node(p, info.graph)
        g_node = sexp_to_node(g, info.graph)

        pn = info.graph.apply(update_rule, p_node, g_node)

        new_model = sexp_to_node(setter_from_getter(p, pn), info.graph)

        return new_model

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
        _cost, dmodel = value_and_grad(cost, "model")(model, inp, target)
        return _cost, update(model, dmodel, update_sgd)

    loss, model_new = step(model, inp, target)

    expected_loss = torch.Tensor([161.05856323242188])
    assert torch.allclose(loss, expected_loss)

    expected_param = torch.Tensor(
        [
            [-0.21953332, -0.31154382, -0.29184943],
            [-0.47497076, -0.39380032, -0.32451797],
            [-0.27165186, -0.96610248, -0.09999254],
            [-0.24826682, 0.73539025, -0.39692938],
        ]
    )

    assert torch.allclose(model_new.W, expected_param)
