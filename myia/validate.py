"""Validate that a graph has been cleaned up and is ready for optimization."""

from .dtype import Array, Tuple, List, Function, Number, Bool, Problem, \
    TypeMeta, TypeType, Class, External, type_cloner
from .dshape import ListShape, TupleShape
from .infer import DEAD
from .ir import manage
from .prim import Primitive, ops as P
from .utils import overload, ErrorPool


class ValidationError(Exception):
    """Error validating a Graph."""


@type_cloner.variant
def _validate_type(self, t: (Class, Problem, External, object)):
    raise ValidationError(f'Illegal type in the graph: {t}')


@overload  # noqa: F811
def _validate_type(self, t: Problem[DEAD]):
    pass


@overload
def _validate_shape(t: Array, shp):
    if not isinstance(shp, tuple):
        raise ValidationError(f'Shape of {t} is {shp}, should be tuple')


@overload  # noqa: F811
def _validate_shape(t: Tuple, shp):
    if not isinstance(shp, TupleShape):
        raise ValidationError(f'Shape of {t} is {shp}, should be TupleShape')
    if len(shp.shape) != len(t.elements):
        raise ValidationError(f'Shape of {t} has wrong length')
    for t2, shp2 in zip(t.elements, shp.shape):
        _validate_shape(t2, shp2)


@overload  # noqa: F811
def _validate_shape(t: List, shp):
    if not isinstance(shp, ListShape):
        raise ValidationError(f'Shape of {t} is {shp}, should be ListShape')
    _validate_shape(t.element_type, shp.shape)


@overload  # noqa: F811
def _validate_shape(t: (Number, TypeType, Bool, Problem[DEAD], Function), shp):
    pass


@overload  # noqa: F811
def _validate_shape(t: TypeMeta, shp):
    return _validate_shape[t](t, shp)


@overload  # noqa: F811
def _validate_shape(t: object, shp):
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
    P.scalar_exp,
    P.scalar_log,
    P.scalar_sin,
    P.scalar_cos,
    P.scalar_tan,
    P.scalar_eq,
    P.scalar_lt,
    P.scalar_gt,
    P.scalar_ne,
    P.scalar_le,
    P.scalar_ge,
    P.scalar_floor,
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
        self.errors = ErrorPool(exc_class=ValidationError)
        self.whitelist = frozenset(whitelist)
        self._run(root)

    def _test(self, node, fn):
        try:
            fn(node)
        except ValidationError as err:
            err.node = node
            self.errors.add(err)

    def _validate_type(self, node):
        return _validate_type(node.type)

    def _validate_oper(self, node):
        if node.is_constant(Primitive):
            if node.value not in self.whitelist:
                raise ValidationError(f'Illegal primitive: {node.value}')

    def _validate_consistency(self, node):
        if node.is_apply():
            expected = Function[[i.type for i in node.inputs[1:]], node.type]
            actual = node.inputs[0].type
            if actual != expected:
                raise ValidationError(
                    f'Function/argument inconsistency: Expected {expected}, '
                    f'Got {actual}.'
                )

    def _validate_shape(self, node):
        return _validate_shape(node.type, node.inferred['shape'])

    def _run(self, root):
        manager = manage(root)
        for node in list(manager.all_nodes):
            self._test(node, self._validate_type)
            self._test(node, self._validate_shape)
            self._test(node, self._validate_oper)
            self._test(node, self._validate_consistency)

        def stringify(err):
            return f'* {err.node} -- {err.args[0]}'

        self.errors.trigger(stringify=stringify)


def validate(root, whitelist=whitelist):
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """
    Validator(root, whitelist)
