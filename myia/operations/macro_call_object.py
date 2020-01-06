"""Implementation of macro `call_object(obj, *args)`.

Receive and call syntax `obj(*args)` and replace it with
relevant myia operations.
"""
from myia.utils.errors import MyiaTypeError

from .. import operations, xtype
from ..lib import (
    DATA,
    AbstractClassBase,
    AbstractError,
    AbstractScalar,
    AbstractType,
    macro,
    type_to_abstract,
)
from ..operations import primitives as P


@macro
async def call_object(info, fn, *n_args):
    """Replace call syntax `fn(*n_args)` with relevant myia operations."""
    fn_node = info.argrefs[0].node
    arg_nodes = (ref.node for ref in info.argrefs[1:])
    g = info.graph
    fn = await fn.get()

    if isinstance(fn, AbstractType):
        # Replace abstract type instantiation with
        # either a cast for abstract scalars, or
        # a make_record for all other cases.
        val = fn.xvalue()
        cls = type_to_abstract(val)
        if isinstance(cls, AbstractScalar):
            typ = cls.xtype()
            if issubclass(typ, xtype.Number):
                newcall = g.apply(P.scalar_cast, *arg_nodes, cls)
            elif typ is xtype.Bool:
                newcall = g.apply(operations.bool, *arg_nodes)
            else:
                raise MyiaTypeError(f'Cannot compile typecast to {typ}')
        else:
            newfn = g.apply(P.partial, P.make_record, val)
            newcall = g.apply(newfn, *arg_nodes)
        return newcall

    elif isinstance(fn, AbstractError):
        raise MyiaTypeError(
            f'Trying to call a function with type '
            f'{fn.xvalue()} {fn.values[DATA] or ""}.'
        )

    elif isinstance(fn, AbstractClassBase):
        newfn = g.apply(operations.getattr, fn_node, '__call__')
        newcall = g.apply(newfn, *arg_nodes)
        return newcall

    else:
        raise MyiaTypeError(f'Myia does not know how to call {fn}')


__operation_defaults__ = {
    'name': 'call_object',
    'registered_name': 'call_object',
    'mapping': call_object,
    'python_implementation': None,
}
