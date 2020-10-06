"""Validate that a graph has been cleaned up and is ready for optimization."""

from types import SimpleNamespace

from ovld import ovld

from . import xtype
from .abstract import (
    ANYTHING,
    DEAD,
    POLY,
    AbstractArray,
    AbstractClass,
    AbstractError,
    AbstractExternal,
    AbstractFunction,
    AbstractJTagged,
    AbstractScalar,
    AbstractType,
    abstract_check,
)
from .operations import Primitive
from .operations.primitives import BackendPrimitive, CompositePrimitive
from .utils import ErrorPool, Partializable


class ValidationError(Exception):
    """Error validating a Graph."""

    def __init__(self, msg, **data):
        """Initialize a ValidationError."""
        super().__init__(msg)
        self.data = data


@abstract_check.variant
def validate_abstract(self, a: (AbstractClass, AbstractJTagged), *, uses):
    """Validate a type."""
    raise ValidationError(f"Illegal type in the graph: {a}", type=a)


@ovld  # noqa: F811
def validate_abstract(self, a: AbstractError, *, uses):
    kind = a.xvalue()
    if kind is DEAD:
        return True
    elif kind is POLY:
        return not any(key == 0 for node, key in uses)
    else:  # pragma: no cover
        # As it turns out, the inferrer now catches this error before we get to
        # validation.
        raise ValidationError(f"Illegal type in the graph: {a}", type=a)


@ovld  # noqa: F811
def validate_abstract(self, a: AbstractArray, *, uses):
    at = a.xtype()
    if at is not ANYTHING and not issubclass(at, xtype.NDArray):
        raise ValidationError(
            f"Illegal array type in the graph: {a.xtype()}", type=a.xtype()
        )
    return True


@ovld  # noqa: F811
def validate_abstract(self, a: AbstractScalar, *, uses):
    if not issubclass(
        a.xtype(),
        (
            xtype.Number,
            xtype.Bool,
            xtype.Nil,
            xtype.EnvType,
            xtype.SymbolicKeyType,
            xtype.UniverseType,
        ),
    ):
        raise ValidationError(
            f"Illegal type in the graph: {a}", type=a
        )  # pragma: no cover
    return True


@ovld  # noqa: F811
def validate_abstract(self, a: (type(None), AbstractExternal), *, uses):
    raise ValidationError(f"Illegal type in the graph: {a}", type=a)


@ovld  # noqa: F811
def validate_abstract(self, a: AbstractType, *, uses):
    return True


@ovld  # noqa: F811
def validate_abstract(self, a: AbstractFunction, *, uses):
    raise ValidationError("Only VirtualFunction is allowed in the final graph")


class NodeValidator:
    """Validate each node in a graph."""

    def __init__(self, resources, errors=None):
        """Initialize a NodeValidator."""
        self.errors = errors or ErrorPool(exc_class=ValidationError)
        self.manager = resources.opt_manager

    def _test(self, node):
        try:
            self.test_node(node)
        except ValidationError as err:
            err.node = node
            node.debug.errors.add(err)
            self.errors.add(err)

    def __call__(self, root):
        """Run on the root graph."""
        for node in list(self.manager.all_nodes):
            self._test(node)

        def stringify(err):
            return f"* {err.node} -- {err.args[0]}"

        self.errors.trigger(stringify=stringify)

    def test_node(self, node):
        """Test whether the node is valid or not."""
        raise NotImplementedError("Override in subclass")


class AbstractValidator(NodeValidator):
    """Test that every node's abstract type is valid."""

    def test_node(self, node):
        """Test that the node's abstract type is valid."""
        return validate_abstract(node.abstract, uses=self.manager.uses[node])


class OperatorValidator(NodeValidator):
    """Test that every node's operator is valid."""

    def test_node(self, node):
        """Test that the node's operator is valid."""
        if node.is_constant(Primitive):
            if not node.is_constant((BackendPrimitive, CompositePrimitive)):
                raise ValidationError(f"Illegal primitive: {node.value}")


class CallValidator(NodeValidator):  # pragma: no cover
    """Test that every operation has a valid type."""

    # This validator works for most tests, but it still has a few issues to
    # work out.

    def __init__(self, *, check_none=True):
        self.check_none = check_none

    def test_node(self, node):
        """Test that the operation for this call has a valid type."""
        if node.is_apply():
            from .abstract import broaden

            fn, *args = node.inputs
            if any(node.abstract is None for node in node.inputs):
                if self.check_none:
                    raise ValidationError(f"None in call")
                else:
                    return
            vfn = fn.abstract
            argv = [broaden(arg) for arg in vfn.args]
            argt = [broaden(arg.abstract) for arg in args]
            if argv != argt:
                raise ValidationError(
                    f"Inconsistent call arguments: {vfn.args}",
                    expected=argv,
                    got=argt,
                )
            if broaden(vfn.output) != broaden(node.abstract):
                raise ValidationError(
                    f"Inconsistent call output: {vfn.output}",
                    expected=broaden(vfn.output),
                    got=broaden(node.abstract),
                )


class MultiValidator(NodeValidator, Partializable):
    """Combine multiple validators."""

    def __init__(self, validators, resources):
        """Initialize the MultiValidator."""
        super().__init__(resources=resources)
        self.resources = resources
        self.validators = [
            v(resources=resources, errors=self.errors) for v in validators if v
        ]

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
        validators=[
            OperatorValidator,
            AbstractValidator,
            # CallValidator,
        ],
        resources=SimpleNamespace(opt_manager=root.manager),
    )
    mv(root)


__all__ = [
    "AbstractValidator",
    "MultiValidator",
    "NodeValidator",
    "OperatorValidator",
    "ValidationError",
    "validate",
    "validate_abstract",
]
