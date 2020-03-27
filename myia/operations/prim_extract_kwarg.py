"""Definitions for the primitive `extract_kwarg`."""

from ..lib import Inferrer, standard_prim
from . import primitives as P


@standard_prim(P.extract_kwarg)
class _ExtractKwArgInferrer(Inferrer):
    """Infer the return type of primitive `extract_kwarg`."""

    async def normalize_args(self, args):
        return args

    async def infer(self, engine, key, kwarg):
        assert key.xvalue() is kwarg.key
        return kwarg.argument


__operation_defaults__ = {
    "name": "extract_kwarg",
    "registered_name": "extract_kwarg",
    "mapping": P.extract_kwarg,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "extract_kwarg",
    "registered_name": "extract_kwarg",
    "type": "inference",
    "python_implementation": None,
    "inferrer_constructor": _ExtractKwArgInferrer,
    "grad_transform": None,
}
