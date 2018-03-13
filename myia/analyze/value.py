"""Value inference."""
from typing import Dict, Callable
from .graph import Plugin

from myia import primops as P
from myia.primops import Primitive
from myia.utils import Named
from myia.anf_ir import Constant, Apply, Parameter, Graph
from myia.py_implementations import implementations as py_implementations

ESTIMATORS: Dict[Primitive, Callable] = dict()


NO_VALUE = Named('NO_VALUE')
NOT_CONSTANT = Named('NOT_CONSTANT')


def register_estimator(prim: Primitive):
    """Register an estimator for a primitive.

    This serves to estimate the value of a primitive in the present of
    incomplete information (not all arguments are constants).  In some
    cases we can know the value (like mul(0, x) -> 0).  If you
    can't do anything unless you know all the arguments don't
    implement this.
    """
    def deco(e: Callable):
        assert prim not in ESTIMATORS
        ESTIMATORS[prim] = e
        return e
    return deco


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

            if fn in (NO_VALUE, NOT_CONSTANT):
                return fn

            if isinstance(fn, Primitive):
                return self.eval_primitive(fn, args)

            elif isinstance(fn, Graph):
                return self.eval_graph(fn, args)

            else:
                raise TypeError("unknown callable type")  # pragma: no cover

    def eval_primitive(self, fn: Primitive, args):
        """Compute the value for a primitive call."""
        if all(a not in (NO_VALUE, NOT_CONSTANT) for a in args):
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
