"""Definitions for the primitive `composite_full`."""

from ..lib import (
    SHAPE,
    TYPE,
    VALUE,
    AbstractArray,
    AbstractScalar,
    AbstractType,
    abstract_array,
    core,
    distribute,
    force_pending,
    scalar_cast,
    standard_prim,
    u64tup_typecheck,
)
from ..xtype import NDArray
from . import primitives as P


@core
def pyimpl_composite_full(shape, fill_value, abstract_scalar_type):
    """Implement `composite_full`."""
    scalar_value = scalar_cast(fill_value, abstract_scalar_type)
    return distribute(
        P.scalar_to_array(scalar_value, abstract_array(shape, scalar_value)),
        shape,
    )


@standard_prim(P.composite_full)
async def infer_composite_full(
    self,
    engine,
    shape: u64tup_typecheck,
    fill_value: AbstractScalar,
    dtype: AbstractType,
):
    """Infer the return type of primitive `composite_full`."""
    return AbstractArray(
        AbstractScalar(
            {
                TYPE: await force_pending(dtype.element.xtype()),
                VALUE: fill_value.xvalue(),
            }
        ),
        {
            SHAPE: tuple(
                self.require_constant(e, argnum=f'"0:shape[{edx}]"')
                for edx, e in enumerate(shape.elements)
            ),
            TYPE: NDArray,
        },
    )


__operation_defaults__ = {
    "name": "composite_full",
    "registered_name": "composite_full",
    "mapping": P.composite_full,
    "python_implementation": pyimpl_composite_full,
}


__primitive_defaults__ = {
    "name": "composite_full",
    "registered_name": "composite_full",
    "type": "composite",
    "python_implementation": pyimpl_composite_full,
    "inferrer_constructor": infer_composite_full,
    "grad_transform": None,
}
