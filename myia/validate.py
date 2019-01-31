"""Validate that a graph has been cleaned up and is ready for optimization."""

from .dtype import Array, Tuple, List, Function, Number, Bool, Problem, \
    TypeMeta, TypeType, Class, External, EnvType, SymbolicKeyType, \
    JTagged, type_cloner, ismyiatype
from .infer import DEAD
from .ir import manage
from .prim import Primitive, ops as P
from .utils import overload, ErrorPool
from .abstract.base import abstract_clone, \
    AbstractClass, AbstractJTagged, AbstractFunction, AbstractScalar, \
    TYPE, VALUE, AbstractType


class ValidationError(Exception):
    """Error validating a Graph."""


@abstract_clone.variant
def validate_abstract(self, a: (AbstractClass, AbstractJTagged)):
    raise ValidationError(f'Illegal type in the graph: {a}')


@overload
def validate_abstract(self, a: AbstractScalar):
    t = a.values[TYPE]
    if ismyiatype(t, (Problem, External)):
        raise ValidationError(f'Illegal type in the graph: {a}')


@overload
def validate_abstract(self, a: AbstractType):
    t = a.values[VALUE]
    if ismyiatype(t, Problem):
        if t.kind is not DEAD:
            raise ValidationError(f'Illegal type in the graph: {a}')


@overload
def validate_abstract(self, a: type(None)):
    raise ValidationError(f'Illegal type in the graph: {a}')


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
    P.bool_eq,
    # P.typeof,
    # P.hastype,
    P.make_tuple,
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
    P.array_to_scalar,
    P.broadcast_shape,
    P.invert_permutation,
    P.shape,
    P.array_map,
    P.array_scan,
    P.array_reduce,
    P.distribute,
    P.reshape,
    P.transpose,
    P.dot,
    P.switch,
    P.return_,
    # P.list_map,
    P.list_append,
    P.make_list,
    P.identity,
    # P.resolve,
    P.partial,
    # P.make_record,
    P.env_getitem,
    P.env_setitem,
    P.env_add,
    # P.J,
    # P.Jinv,
    P.scalar_cast,
})


class Validator:
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """

    def __init__(self, root, whitelist, _validate_abstract=validate_abstract):
        """Initialize and run the Validator."""
        self.errors = ErrorPool(exc_class=ValidationError)
        self.whitelist = frozenset(whitelist)
        self._validate_abstract_fn = _validate_abstract
        self._run(root)

    def _test(self, node, fn):
        try:
            fn(node)
        except ValidationError as err:
            err.node = node
            self.errors.add(err)

    def _validate_oper(self, node):
        if node.is_constant(Primitive):
            if node.value not in self.whitelist:
                raise ValidationError(f'Illegal primitive: {node.value}')

    def _validate_abstract(self, node):
        return self._validate_abstract_fn(node.abstract)

    def _run(self, root):
        manager = manage(root)
        for node in list(manager.all_nodes):
            self._test(node, self._validate_abstract)
            self._test(node, self._validate_oper)

        def stringify(err):
            return f'* {err.node} -- {err.args[0]}'

        self.errors.trigger(stringify=stringify)


def validate(root, whitelist=whitelist, validate_abstract=validate_abstract):
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must belong to the whitelist.
    """
    Validator(root, whitelist, validate_abstract)
