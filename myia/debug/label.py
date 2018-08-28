"""Utilities to generate or map labels for nodes and graphs."""


from ..info import DebugInfo
from ..ir.anf import ANFNode, Graph
from ..prim import Primitive
from ..utils import Named, Namespace


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
    'specialized': '+'
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
                 relation_symbols={}):
        """Initialize a NodeLabeler."""
        self.function_in_node = function_in_node
        self.relation_symbols = relation_symbols

    def combine_relation(self, name, relation):
        """Combine a name and a relation in a single string."""
        rel = self.relation_symbols.get(relation, f'{relation}:')
        if rel is None:
            return None
        if rel:
            return f'{rel}{name or ""}'
        else:
            return name

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

    def name(self, node, force=False):
        """Return a node's name."""
        if isinstance(node, DebugInfo):
            if node.name:
                return node.name
            elif node.about:
                return self.combine_relation(
                    self.name(node.about.debug, force),
                    node.about.relation
                )
            elif force:
                return f'#{node.id}'
            else:
                return None
        else:
            return self.name(node.debug, force)

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
            if v is None or v == ():
                return repr(v)
            elif isinstance(v, (int, float, str, Named, Namespace)):
                return repr(v)
            elif isinstance(v, Primitive):
                return v.name
            elif isinstance(v, CosmeticPrimitive):
                return v.label
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


short_labeler = NodeLabeler(
    relation_symbols=short_relation_symbols
)


def label(x):
    """Return an informative textual label for a node."""
    if isinstance(x, Primitive):
        return x.name
    elif isinstance(x, (ANFNode, Graph, DebugInfo)):
        return short_labeler.name(x)
    else:
        return repr(x)
