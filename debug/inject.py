"""Utility injecting functionality, only use for debugging."""

import os
import sys

import breakword
from buche import buche

# Load custom hrepr methods for Graph etc.
from . import gprint  # noqa

_log = []


def ibuche(*args, **kwargs):
    _log.append(args)
    buche(*args, interactive=True, **kwargs)


def _lwlog(*objs):
    if os.environ.get('BUCHE'):
        ibuche(*objs)
    else:
        print(*objs)


breakword.set_default_logger(_lwlog)


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
    'logword': breakword.log,
    'getword': breakword.word,
    'afterword': breakword.after,
    'breakword': breakword.brk,
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
