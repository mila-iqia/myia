"""Utilities to generate or map labels for nodes and graphs."""


from myia.info import DebugInfo
from myia.anf_ir import Graph
from myia.anf_ir_utils import \
    is_constant, is_constant_graph, is_parameter, is_special
from myia.prim import Primitive


short_relation_symbols = {
    'copy': '',
    'phi': 'Φ',
    'if_true': '✓',
    'if_false': '✗',
    'if_after': '↓',
    'while_header': '⤾',
    'while_body': '⥁',
    'while_after': '↓'
}


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
        return f'{rel}{name}'

    def const_fn(self, node):
        """
        Return name of function, if constant.

        Given an `Apply` node of a constant function, return the
        name of that function, otherwise return None.
        """
        fn = node.inputs[0] if node.inputs else None
        if fn and is_constant(fn):
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
                    self.name(node.about.debug),
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
        elif is_constant_graph(node):
            return self.name(node.value.debug,
                             True if force is None else force)
        elif is_constant(node):
            v = node.value
            if isinstance(v, (int, float, str)):
                return repr(v)
            elif isinstance(v, Primitive):
                return v.name
            else:
                class_name = v.__class__.__name__
                return f'{self.label(node.debug, True)}:{class_name}'
        elif is_special(node):
            return f'{node.special}'
        elif is_parameter(node):
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
