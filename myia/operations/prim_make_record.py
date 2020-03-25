"""Definitions for the primitive `make_record`."""

from .. import lib
from ..lib import (
    AbstractADT,
    MyiaTypeError,
    broaden,
    standard_prim,
    type_to_abstract,
    typecheck,
)
from . import primitives as P


@standard_prim(P.make_record)
async def infer_make_record(self, engine, _cls: lib.AbstractType, *elems):
    """Infer the return type of primitive `make_record`."""
    cls = _cls.element
    cls = type_to_abstract(cls)
    expected = list(cls.attributes.items())
    if len(expected) != len(elems):
        raise MyiaTypeError(
            f"{cls.tag.__qualname__} expects {len(expected)} fields "
            f"but got {len(elems)} instead."
        )
    for (name, t), elem in zip(expected, elems):
        if not typecheck(t, elem):
            raise MyiaTypeError(
                f"{cls.tag.__qualname__} expects field `{name}` "
                f"to have type {elem} but got {t}"
            )

    wrap = broaden if type(cls) is AbstractADT else None

    return type(cls)(
        cls.tag,
        {
            name: wrap(elem) if wrap else elem
            for (name, _), elem in zip(expected, elems)
        },
        constructor=cls.constructor,
    )


__operation_defaults__ = {
    "name": "make_record",
    "registered_name": "make_record",
    "mapping": P.make_record,
    "python_implementation": None,
}


__primitive_defaults__ = {
    "name": "make_record",
    "registered_name": "make_record",
    "type": "inference",
    "python_implementation": None,
    "inferrer_constructor": infer_make_record,
    "grad_transform": None,
}
