"""Value inference."""
from typing import Dict, Callable

from myia.prim import (Primitive, ops as P,
                       implementations as py_implementations)
from myia.utils import Named, Registry
from myia.anf_ir import Constant, Apply, Parameter, Graph

from .graph import Plugin

ESTIMATORS: Registry[Primitive, Callable] = Registry()
register_estimator = ESTIMATORS.register


NO_VALUE = Named('NO_VALUE')


# SCCP
class ValuePlugin(Plugin):
    """Implementation of SCCP (Sparse Conditional Constant Propagation).

    This has some trouble with loops and such since those are
    implemented using recursive calls and we don't handle that.

    It does deal with conditionals ok.
    """

    NAME = "value"

    def __init__(self, *, implementations: Dict[Primitive, Callable] = None,
                 estimators: Dict[Primitive, Callable] = None) -> None:
        """Create a ValuePlugin."""
        if implementations is None:
            implementations = py_implementations
        self._implementations = implementations
        if estimators is None:
            estimators = ESTIMATORS
        self._estimators = estimators

    def visit(self, fn: Callable, value):
        """Nothing to visit."""
        raise self.analyzer.DU.VisitError  # pragma: no cover

    def on_attach(self):
        """Add shortcuts."""
        # Don't explicitely depend on TypePlugin since that will create a cycle

        self.analyzer.add_shortcut('get_value', self.get_value)

    def get_value(self, node):
        """Shortcut to get the value associated with a node."""
        return self.analyzer._info_map[node][self.NAME]

    def on_node(self, node):
        """Compute the value for the node."""
        if isinstance(node, Constant):
            return node.value

        elif isinstance(node, Parameter):
            return NO_VALUE

        elif isinstance(node, Apply):
            args = tuple(self.get_value(i) for i in node.inputs)
            fn = args[0]
            args = args[1:]

            if fn is NO_VALUE:
                return fn

            if isinstance(fn, Primitive):
                return self.eval_primitive(fn, args)

            elif isinstance(fn, Graph):
                return self.eval_graph(fn, args)

            else:
                raise TypeError("unknown callable type")  # pragma: no cover

    def eval_primitive(self, fn: Primitive, args):
        """Compute the value for a primitive call."""
        if all(a is not NO_VALUE for a in args):
            if fn == P.if_:
                if args[0]:
                    return self.eval_graph(args[1], ())
                else:
                    return self.eval_graph(args[2], ())
            else:
                return self._implementations[fn](*args)
        else:
            return self._estimator(fn)(*args)

    def eval_graph(self, fn: Graph, args):
        """Compute the value for a graph call."""
        # this will handle no-args and constant-return functions ok
        # functions that actually use their arguments should return
        # NO_VALUE here which is what we want.
        try:
            return self.get_value(fn.return_)
        except KeyError:
            # This will happen for recursion
            return NO_VALUE

    def _estimator(self, fn):
        if fn in self._estimators:
            return self._estimators[fn]
        else:
            return lambda *v: NO_VALUE


@register_estimator(P.mul)
def _est_mul(x, y):
    if x == 0 or y == 0:
        return 0
    else:
        return NO_VALUE
