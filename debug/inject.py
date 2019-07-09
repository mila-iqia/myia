"""Utility injecting functionality, only use for debugging."""

import sys

from buche import buche

# Load custom hrepr methods for Graph etc.
from . import gprint  # noqa
from .logword import ibuche, logword, afterword, breakword


def bucheg(graph, **kwargs):
    from myia.ir import ANFNode

    def ttip(node):
        if isinstance(node, ANFNode):
            return node.abstract

    kw = dict(node_tooltip=ttip, function_in_node=True,
              graph_width='95vw', graph_height='95vh')
    kw.update(kwargs)
    buche(graph, **kw)


suite = {
    'buche': buche,
    'bucheg': bucheg,
    'ibuche': ibuche,
    'Subgraph': gprint.Subgraph,
    'logword': logword,
    'afterword': afterword,
    'breakword': breakword,
}


def inject(**utilities):
    """Inject all utilities in the globals of every module."""
    for name, module in list(sys.modules.items()):
        glob = vars(module)
        for key, value in utilities.items():
            if key not in glob:
                try:
                    glob[key] = value
                except TypeError:
                    pass


def inject_suite():
    """Inject default utilities in the globals of every module."""
    inject(**suite)
