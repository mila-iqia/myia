"""Implementation of the 'getattr' operation."""

from .. import lib, operations
from ..lib import (
    ANYTHING,
    Constant,
    Graph,
    InferenceError,
    MyiaAttributeError,
    MyiaTypeError,
    Pending,
    build_value,
    macro,
)
from . import primitives as P


async def _resolve_case(resources, data, data_t, item_v):
    mmap = resources.method_map
    is_cls = isinstance(data, lib.AbstractClassBase)

    if data_t is None or data_t is type:
        mro = ()
    else:
        mro = data_t.mro()

    for t in mro:
        if t in mmap:
            mmap_t = mmap[t]
            if item_v in mmap_t:
                method = mmap_t[item_v]
                return ('method', method)
    else:
        if is_cls:
            if item_v in data.attributes:
                return ('field', item_v)
            elif hasattr(data_t, item_v):
                return ('method', getattr(data_t, item_v))
            else:
                return ('no_method',)
        else:
            return ('static',)


async def _attr_case(info, data, attr_v):
    data_t = data.xtype()
    if attr_v is ANYTHING:
        raise InferenceError(
            'The value of the attribute could not be inferred.'
        )
    elif not isinstance(attr_v, str):  # pragma: no cover
        raise MyiaTypeError(
            f'Argument to getattr must be a string, not {attr_v}.'
        )

    resources = info.engine.resources
    if isinstance(data_t, Pending):
        return await lib.find_coherent_result(
            data_t,
            lambda t: _resolve_case(resources, data, t, attr_v)
        )
    else:
        return await _resolve_case(resources, data, data_t, attr_v)


async def _union_make(info, op):
    # Helper for hasattr/getattr on unions.
    # op(x :: U(T1, T2, T3), attr) becomes:
    # (op(T1(x), attr) if hastype(x, T1)
    #  else (op(T2(x), attr) if hastype(x, T2)
    #        else op(T3(x), attr)))
    r_data, r_attr = info.argrefs
    data, attr = await info.abstracts()
    currg = info.graph
    opts = await lib.force_pending(data.options)
    for i, opt in enumerate(opts):
        last = (i == len(opts) - 1)
        if last:
            falseg = None
            cast = currg.apply(P.unsafe_static_cast, r_data.node, opt)
            out = currg.apply(op, cast, r_attr.node)
        else:
            trueg = Graph()
            falseg = Graph()
            cond = currg.apply(P.hastype, r_data.node, opt)
            cast = trueg.apply(P.unsafe_static_cast, r_data.node, opt)
            trueg.output = trueg.apply(op, cast, r_attr.node)
            info.engine.mng.add_graph(trueg)
            out = currg.apply(P.switch, cond, trueg, falseg)
            out = currg.apply(out)
        if currg is info.graph:
            rval = out
        else:
            currg.output = out
            info.engine.mng.add_graph(currg)
        currg = falseg
    return rval


@macro
async def getattr_(info, r_data, r_attr):
    """Get an attribute from an object."""
    data, attr = await info.abstracts()
    g = info.graph

    if isinstance(data, lib.AbstractUnion):
        return await _union_make(info, operations.getattr)

    attr_v = build_value(attr, default=ANYTHING)
    case, *args = await _attr_case(info, data, attr_v)

    if case == 'field':
        # Get field from Class
        return g.apply(P.record_getitem, r_data.node, attr_v)

    elif case == 'method':
        method, = args
        if isinstance(method, property):
            return g.apply(method.fget, r_data.node)
        else:
            return g.apply(P.partial, method, r_data.node)

    elif case == 'no_method':
        msg = f"object of type {data} has no attribute '{attr_v}'"
        raise MyiaAttributeError(msg)

    else:
        # Module or static namespace
        data_v = build_value(data, default=ANYTHING)
        if data_v is ANYTHING:
            raise InferenceError(
                f"Could not resolve attribute '{attr_v}' on "
                f"object of type {data}."
            )
        try:
            raw = getattr(data_v, attr_v)
        except AttributeError as e:
            raise MyiaAttributeError(str(e))
        except Exception as e:  # pragma: no cover
            raise InferenceError(f'Unexpected error in getter: {e!r}')
        return Constant(raw)


__operation_defaults__ = {
    'name': 'getattr',
    'registered_name': 'getattr',
    'mapping': getattr_,
    'python_implementation': getattr,
}
