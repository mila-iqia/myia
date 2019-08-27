"""Utilities to generate or map labels for nodes and graphs."""


from types import FunctionType

from ..info import DebugInfo
from ..ir import MetaGraph
from ..ir.anf import ANFNode, Graph
from ..prim import Primitive
from ..utils import EnvInstance, Named, Namespace, SymbolicKeyInstance

short_relation_symbols = {
    'copy': '',
    'cosmetic': '',
    'phi': 'Φ',
    'iterator': '*',
    'fv': '⤋',
    'if_true': '✓',
    'if_false': '✗',
    'if_after': '↓',
    'while_header': '⤾',
    'while_body': '⥁',
    'while_after': '↓',
    'for_header': '⤾',
    'for_body': '⥁',
    'for_after': '↓',
    'specialized': '+',
    'equiv': '',
    'grad_fprop_app': '',
    'grad_bprop_app': '▼',
    'grad_fprop': '▶',
    'grad_bprop': '◀',
    'grad_sens': '∇',
    'opt': '',
}


class CosmeticPrimitive:
    """Primitive with no other utility than printing pretty."""

    def __init__(self, label):
        """Initialize a CosmeticPrimitive."""
        self.label = label


class NodeLabeler:
    """Utility to label a node."""

    def __init__(self,
                 function_in_node=True,
                 relation_symbols={},
                 default_name=lambda dbg: f'#{dbg.id}'):
        """Initialize a NodeLabeler."""
        self.function_in_node = function_in_node
        self.relation_symbols = relation_symbols
        self.default_name = default_name

    def _root_name(self, node, force):
        if isinstance(node, DebugInfo):
            if node.name:
                return node.name, []
            elif node.about:
                root_name, relations = self._root_name(node.about.debug, force)
                relations.append(node.about.relation)
                return root_name, relations
            elif force:
                return self.default_name(node), []
            else:
                return None, []
        else:
            return self._root_name(node.debug, force)

    def combine_relations(self, root_name, relations):
        """Combine a name and a list of relations in a single string."""
        if root_name is None:
            return None

        ids = [r for r in relations if isinstance(r, int)]
        relations = [r for r in relations if not isinstance(r, int)]

        if ids:
            relations.append(ids[-1])

        tags = [self.relation_symbols.get(r, f'{r}:')
                for r in reversed(relations)]

        return ''.join(tags) + root_name

    def name(self, node, force=False):
        """Return a node's name."""
        root_name, relations = self._root_name(node, force)
        return self.combine_relations(root_name, relations)

    def label(self, node, force=None, fn_label=None):
        """Label a node."""
        if isinstance(node, DebugInfo):
            return self.name(node, True if force is None else force)
        elif isinstance(node, Graph):
            return self.name(node.debug,
                             True if force is None else force)
        elif node.is_constant_graph():
            return self.name(node.value.debug,
                             True if force is None else force)
        elif node.is_constant():
            v = node.value
            if v is None or (isinstance(v, tuple) and v == ()):
                return repr(v)
            elif isinstance(v, (int, float, str, Named, Namespace)):
                return repr(v)
            elif isinstance(v, FunctionType):
                return f'{v.__name__}::function'
            elif isinstance(v, (Primitive, MetaGraph)):
                return v.name
            elif isinstance(v, CosmeticPrimitive):
                return v.label
            elif isinstance(v, SymbolicKeyInstance):
                return f'{self.name(v.node)}::Node'
            elif isinstance(v, EnvInstance):
                if len(v):
                    return 'EnvType(...)'
                else:
                    return "∅"
            else:
                class_name = v.__class__.__name__
                s = str(v)
                if len(s) > 10:
                    s = f'{s[:10]}...'
                return f'{s}::{class_name}'
        elif node.is_special():
            return f'{node.special}'
        elif node.is_parameter():
            return self.label(node.debug, True)
        else:
            lbl = ''
            if self.function_in_node:
                if fn_label is None:
                    fn_label = self.const_fn(node)
                if fn_label:
                    lbl = fn_label

            name = self.name(node, force)
            if name:
                if lbl:
                    lbl += f'→{name}'
                else:
                    lbl = name
            return lbl or '·'

    def const_fn(self, node):
        """Return name of function, if constant.

        Given an `Apply` node of a constant function, return the
        name of that function, otherwise return None.
        """
        fn = node.inputs[0] if node.inputs else None
        if fn and fn.is_constant():
            return self.label(fn, False)
        else:
            return None


short_labeler = NodeLabeler(
    relation_symbols=short_relation_symbols
)


default_labeler = NodeLabeler(
    relation_symbols=short_relation_symbols,
    default_name=lambda dbg: dbg.debug_name
)


def label(x, labeler=default_labeler):
    """Return an informative textual label for a node."""
    if isinstance(x, Primitive):
        return x.name
    elif isinstance(x, (ANFNode, Graph, DebugInfo)):
        return labeler.name(x, True)
    else:
        return repr(x)
