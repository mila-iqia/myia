"""Validate that a graph has been cleaned up and is ready for optimization."""

from .dtype import Tuple, List, Function, Type, Number, Bool, Problem, TypeMeta
from .ir import manage
from .prim import Primitive, ops as P
from .specialize import DEAD
from .utils import overload


class ValidationError(Exception):
    """Error validating a Graph."""


@overload
def _validate_type(t: Tuple):
    for t2 in t.elements:
        _validate_type(t2)


@overload  # noqa: F811
def _validate_type(t: List):
    _validate_type(t.element_type)


@overload  # noqa: F811
def _validate_type(t: Function):
    for t2 in t.arguments:
        _validate_type(t2)
    _validate_type(t.retval)


@overload  # noqa: F811
def _validate_type(t: (Number, Bool, Problem[DEAD])):
    pass


@overload  # noqa: F811
def _validate_type(t: Type):
    raise ValidationError(f'Illegal type in the graph: {t}')


@overload  # noqa: F811
def _validate_type(t: TypeMeta):
    return _validate_type[t](t)


@overload  # noqa: F811
def _validate_type(t: object):
    raise ValidationError(f'Illegal type in the graph: {t}')


# All the legal operations are listed here.
# Illegal ones are commented out.
whitelist = frozenset({
    P.scalar_add,
    P.scalar_sub,
    P.scalar_mul,
    P.scalar_div,
    P.scalar_mod,
    P.scalar_pow,
    P.scalar_uadd,
    P.scalar_usub,
    P.scalar_eq,
    P.scalar_lt,
    P.scalar_gt,
    P.scalar_ne,
    P.scalar_le,
    P.scalar_ge,
    P.bool_not,
    P.bool_and,
    P.bool_or,
    # P.typeof,
    # P.hastype,
    P.make_tuple,
    P.tail,
    P.tuple_getitem,
    P.list_getitem,
    P.array_getitem,
    P.tuple_setitem,
    P.list_setitem,
    P.array_setitem,
    # P.getattr,
    # P.setattr,
    P.tuple_len,
    P.list_len,
    P.array_len,
    P.scalar_to_array,
    P.broadcast_shape,
    P.shape,
    P.array_map,
    P.array_scan,
    P.array_reduce,
    P.distribute,
    P.reshape,
    P.dot,
    P.if_,
    P.switch,
    P.return_,
    P.list_map,
    P.identity,
    # P.resolve,
    P.partial,
    # P.make_record,
})


class Validator:
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """

    def __init__(self, root, whitelist):
        """Initialize and run the Validator."""
        self.errors = []
        self.whitelist = frozenset(whitelist)
        self._run(root)

    def _test(self, node, fn):
        try:
            fn(node)
        except ValidationError as err:
            err.node = node
            self.errors.append(err)

    def _validate_type(self, node):
        return _validate_type(node.type)

    def _validate_oper(self, node):
        if node.is_constant(Primitive):
            if node.value not in self.whitelist:
                raise ValidationError(f'Illegal primitive: {node.value}')

    def _validate_consistency(self, node):
        if node.is_apply():
            expected = Function[[i.type for i in node.inputs[1:]], node.type]
            # _validate_consistency(node.inputs)
            actual = node.inputs[0].type
            if actual != expected:
                raise ValidationError(
                    f'Function/argument inconsistency: Expected {expected}, '
                    f'Got {actual}.'
                )

    def _run(self, root):
        manager = manage(root)
        for node in list(manager.all_nodes):
            self._test(node, self._validate_type)
            self._test(node, self._validate_oper)
            self._test(node, self._validate_consistency)
        if self.errors:
            errset = {f'* {err.args[0]}' for err in self.errors}
            errlist = "\n".join(sorted(errset))
            err = ValidationError(f'The graph is not valid:\n\n{errlist}')
            err.errors = self.errors
            raise err


def validate(root, whitelist=whitelist):
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """
    Validator(root, whitelist)
