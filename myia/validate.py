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
    AbstractType,
    abstract_check,
)
from .ir import manage
from .operations import Primitive
from .operations.primitives import BackendPrimitive
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
                       xtype.EnvType, xtype.SymbolicKeyType,
                       xtype.UniverseType)):
        raise ValidationError(
            f'Illegal type in the graph: {a}'
        )   # pragma: no cover


@overload  # noqa: F811
def validate_abstract(self, a: (type(None), AbstractExternal), uses):
    raise ValidationError(f'Illegal type in the graph: {a}')


@overload  # noqa: F811
def validate_abstract(self, a: AbstractType, uses):
    return True


class NodeValidator:
    """Validate each node in a graph."""

    def _test(self, node):
        try:
            self.test_node(node)
        except ValidationError as err:
            err.node = node
            node.debug.errors.add(err)
            self.errors.add(err)

    def setup(self, root, errors=None, manager=None):
        """Set the error pool and the manager."""
        self.errors = errors or ErrorPool(exc_class=ValidationError)
        self.manager = manager or manage(root)

    def run(self, root):
        """Run on the root graph."""
        self.setup(root)
        for node in list(self.manager.all_nodes):
            self._test(node)

        def stringify(err):
            return f'* {err.node} -- {err.args[0]}'

        self.errors.trigger(stringify=stringify)

    def test_node(self, node):
        """Test whether the node is valid or not."""
        raise NotImplementedError('Override in subclass')


class AbstractValidator(NodeValidator):
    """Test that every node's abstract type is valid."""

    def test_node(self, node):
        """Test that the node's abstract type is valid."""
        return validate_abstract(node.abstract, self.manager.uses[node])


class OperatorValidator(NodeValidator):
    """Test that every node's operator is valid."""

    def test_node(self, node):
        """Test that the node's operator is valid."""
        if node.is_constant(Primitive):
            if not node.is_constant(BackendPrimitive):
                raise ValidationError(f'Illegal primitive: {node.value}')


class MultiValidator(NodeValidator):
    """Combine multiple validators."""

    def __init__(self, *validators):
        """Initialize the MultiValidator."""
        self.validators = [v for v in validators if v]

    def setup(self, root):
        """Set up all validators."""
        super().setup(root)
        for v in self.validators:
            v.setup(root, errors=self.errors, manager=self.manager)

    def test_node(self, node):
        """Test the node through every validator."""
        for v in self.validators:
            v.test_node(node)


def validate(root):
    """Verify that g is properly type-specialized.

    Every node of each graph must have a concrete type in its type attribute,
    every application must be compatible with its argument types, and every
    primitive must be a BackendPrimitive.
    """
    mv = MultiValidator(
        OperatorValidator(),
        AbstractValidator(),
    )
    mv.run(root)


__all__ = [
    'AbstractValidator',
    'MultiValidator',
    'NodeValidator',
    'OperatorValidator',
    'ValidationError',
    'validate',
    'validate_abstract',
]
