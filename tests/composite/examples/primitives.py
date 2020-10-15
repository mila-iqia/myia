from myia.abstract import ANYTHING
from myia.abstract.infer import standard_prim
from myia.operations.utils import CompositePrimitive
from myia.opt.dde import regvprop

from . import prim_composite_full, prim_composite_simple

composite_full = CompositePrimitive(name="composite_full", defaults={})
composite_simple = CompositePrimitive(name="composite_simple", defaults={})

composite_full.set_defaults(
    {
        "name": "composite_full",
        "registered_name": "composite_full",
        "type": "composite",
        "python_implementation": prim_composite_full.pyimpl_composite_full,
        "inferrer_constructor": standard_prim(composite_full)(
            prim_composite_full.infer_composite_full
        ),
        "grad_transform": None,
    }
)
composite_simple.set_defaults(
    {
        "name": "composite_simple",
        "registered_name": "composite_simple",
        "type": "composite",
        "python_implementation": prim_composite_simple.pyimpl_composite_simple,
        "inferrer_constructor": standard_prim(composite_simple)(
            prim_composite_simple.infer_composite_simple
        ),
        "grad_transform": None,
    }
)


@regvprop(
    composite_full, composite_simple,
)
def _vprop_generic(vprop, need, inputs, out):
    for inp in inputs:
        vprop.add_need(inp, ANYTHING)
    vprop.add_value(out, need, ANYTHING)
