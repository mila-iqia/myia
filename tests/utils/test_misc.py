import pytest

from myia.utils.misc import ModuleNamespace, Named, Namespace


def test_str():
    assert str(Named("TEST")) == "TEST"


t_namespace = Namespace(
    "test", dict(e=5, b=0), dict(a=33, d=4), dict(a=1, b=2, c=3)
)


def test_namespace():
    assert str(t_namespace) == ":test"


def test_namespace_contains():
    assert "a" in t_namespace
    assert "b" in t_namespace
    assert "c" in t_namespace
    assert "d" in t_namespace
    assert "e" in t_namespace
    assert "f" not in t_namespace


def test_namespace_getitem():
    assert t_namespace["a"] == 33
    assert t_namespace["b"] == 0
    assert t_namespace["c"] == 3
    assert t_namespace["d"] == 4
    assert t_namespace["e"] == 5

    with pytest.raises(NameError):
        t_namespace["f"]


def test_module_namespace():
    m_ns = ModuleNamespace("builtins")

    assert str(m_ns) == ":builtins"
    assert "str" in m_ns
    assert "_test_method_name_123456" not in m_ns
