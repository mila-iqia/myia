import pytest

try:
    import jax  # noqa: F401
except ImportError:
    raise pytest.skip("jax is not available", allow_module_level=True)

from myia.jax import checked_jit


def test_dec():
    @checked_jit
    def f(a):
        return a


def test_dec2():
    @checked_jit(static_argnames=("a",))
    def f(a, b):
        return a + b
