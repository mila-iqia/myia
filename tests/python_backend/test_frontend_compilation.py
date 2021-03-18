import pytest

import test_graph_compilation
from myia.frontends import activate_frontend  # noqa: E402

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional
activate_frontend("pytorch")


parse_and_compile = test_graph_compilation.parse_and_compile


def test_argmax():
    @parse_and_compile
    def f(x):
        return torch.argmax(x)

    x = torch.randn(2, 3)
    print(f(x))
