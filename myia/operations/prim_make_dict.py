"""Definitions for the primitive `make_dict`."""

from .. import lib
from ..lib import AbstractDict, standard_prim, typecheck
from . import primitives as P


@standard_prim(P.make_dict)
async def infer_make_dict(self, engine, _dct: lib.AbstractType, *values):
    """Infer the return type of primitive `make_dict`."""
    dct = _dct.element
    assert len(dct.entries) == len(values)
    for t, elem in zip(dct.entries.values(), values):
        assert typecheck(t, elem)

    return AbstractDict(
        dict((key, val) for key, val in zip(dct.entries.keys(), values))
    )


__operation_defaults__ = {
    "name": "make_dict",
    "registered_name": "make_dict",
    "mapping": P.make_dict,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "make_dict",
    "registered_name": "make_dict",
    "type": "inference",
    "python_implementation": None,
    "inferrer_constructor": infer_make_dict,
    "grad_transform": None,
}
