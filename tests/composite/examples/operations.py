from myia.operations import Operation

from . import prim_composite_full, prim_composite_simple
from .primitives import composite_full, composite_simple

composite_full = Operation(
    name="composite_full",
    defaults={
        "name": "composite_full",
        "registered_name": "composite_full",
        "mapping": composite_full,
        "python_implementation": prim_composite_full.pyimpl_composite_full,
    },
)
composite_simple = Operation(
    name="composite_simple",
    defaults={
        "name": "composite_simple",
        "registered_name": "composite_simple",
        "mapping": composite_simple,
        "python_implementation": prim_composite_simple.pyimpl_composite_simple,
    },
)
