"""Adapter to combine jax decorators with myia's checked decorator."""
import jax

from .api import CheckedFunction
from .parser import parse
from .utils.info import enable_debug


def checked_jit(
    fun=None,
    *,
    static_argnums=None,
    static_argnames=None,
    device=None,
    backend=None,
    donate_argnums=()
):
    """Adapter to combine @myia.api.checked with @jax.jit.

    See the their respective documentation for behaviour and arguments.
    """
    if fun is None:
        return jax.partial(
            checked_jit,
            static_argnums=static_argnums,
            static_argnames=static_argnames,
            device=device,
            backend=backend,
            donate_argnums=donate_argnums,
        )
    with enable_debug():
        graph = parse(fun)
    fun = jax.jit(
        fun=fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        device=device,
        backend=backend,
        donate_argnums=donate_argnums,
    )
    return CheckedFunction(fun, graph=graph)
