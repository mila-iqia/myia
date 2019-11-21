"""Validate that a graph has been cleaned up and is ready for optimization."""

from . import xtype
from .abstract import (
    DEAD,
    POLY,
    AbstractClass,
    AbstractError,
    AbstractExternal,
    AbstractJTagged,
    AbstractScalar,
    abstract_check,
)
from .ir import manage
from .operations import Primitive, primitives as P
from .utils import ErrorPool, overload


class ValidationError(Exception):
    """Error validating a Graph."""


@abstract_check.variant
def validate_abstract(self, a: (AbstractClass, AbstractJTagged), uses):
    """Validate a type."""
    raise ValidationError(f'Illegal type in the graph: {a}')


@overload  # noqa: F811
def validate_abstract(self, a: AbstractError, uses):
    kind = a.xvalue()
    if kind is DEAD:
        return True
    elif kind is POLY:
        return not any(key == 0 for node, key in uses)
    else:  # pragma: no cover
        # As it turns out, the inferrer now catches this error before we get to
        # validation.
        raise ValidationError(f'Illegal type in the graph: {a}')


@overload  # noqa: F811
def validate_abstract(self, a: AbstractScalar, uses):
    if not issubclass(a.xtype(),
                      (xtype.Number, xtype.Bool, xtype.Nil,
                       xtype.EnvType, xtype.SymbolicKeyType)):
        raise ValidationError(
            f'Illegal type in the graph: {a}'
        )   # pragma: no cover


@overload  # noqa: F811
def validate_abstract(self, a: (type(None), AbstractExternal), uses):
    raise ValidationError(f'Illegal type in the graph: {a}')


# All the legal operations are listed here.
# Illegal ones are commented out.
whitelist = frozenset({
    P.scalar_abs,
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
    P.scalar_tanh,
    P.scalar_eq,
    P.scalar_lt,
    P.scalar_gt,
    P.scalar_ne,
    P.scalar_le,
    P.scalar_ge,
    P.scalar_floor,
    P.scalar_max,
    P.scalar_sign,
    P.bool_not,
    P.scalar_bit_and,
    P.scalar_bit_or,
    P.scalar_bit_xor,
    P.scalar_bit_lshift,
    P.scalar_bit_rshift,
    P.bool_and,
    P.bool_or,
    P.bool_eq,
    P.make_tuple,
    P.tuple_getitem,
    P.array_getitem,
    P.tuple_setitem,
    P.array_setitem,
    # P.record_getitem,
    # P.record_setitem,
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
    P.identity,
    P.partial,
    # P.make_record,
    P.env_getitem,
    P.env_setitem,
    P.env_add,
    # P.J,
    # P.Jinv,
    P.scalar_cast,
    P.array_cast,
    P.hastag,
    P.casttag,
    P.tagged,
    P.unsafe_static_cast,
    P.gather,
    P.scatter,
    P.scatter_add,
    P.argmax,
    P.array_max,
    P.concat,
    P.conv2d,
    P.conv2d_input_grad,
    P.conv2d_weight_grad,
    P.max_pool2d,
    P.max_pool2d_grad,
    P.split,
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
            node.debug.errors.add(err)
            self.errors.add(err)

    def _validate_oper(self, node):
        if node.is_constant(Primitive):
            if node.value not in self.whitelist:
                raise ValidationError(f'Illegal primitive: {node.value}')

    def _validate_abstract(self, node):
        return self._validate_abstract_fn(node.abstract,
                                          self.manager.uses[node])

    def _run(self, root):
        self.manager = manage(root)
        for node in list(self.manager.all_nodes):
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


__all__ = [
    'ValidationError',
    'Validator',
    'validate',
    'validate_abstract',
]
