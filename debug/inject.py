"""Utility injecting functionality, only use for debugging."""

import sys

from buche import buche

# Load custom hrepr methods for Graph etc.
from . import gprint  # noqa


def bucheg(graph, **kwargs):
    from myia.ir import ANFNode

    def ttip(node):
        if isinstance(node, ANFNode):
            return node.abstract
    buche(graph, node_tooltip=ttip, function_in_node=True,
          graph_width='95vw', graph_height='95vh',
          **kwargs)


_log = []


def ibuche(*args, **kwargs):
    _log.append(args)
    buche(*args, interactive=True, **kwargs)


suite = {
    'buche': buche,
    'bucheg': bucheg,
    'ibuche': ibuche,
    'Subgraph': gprint.Subgraph,
}


def inject(**utilities):
    """Inject all utilities in the globals of every module."""
    for name, module in list(sys.modules.items()):
        glob = vars(module)
        for key, value in utilities.items():
            if key not in glob:
                try:
                    glob[key] = value
                except TypeError as e:
                    pass


def inject_suite():
    """Inject default utilities in the globals of every module."""
    inject(**suite)
