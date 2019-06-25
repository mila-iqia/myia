"""Tools to print a traceback for an error in Myia."""

import ast
import sys
import warnings
import prettyprinter as pp

from ..abstract import InferenceError, Reference, data, format_abstract, \
    pretty_struct
from ..parser import MyiaSyntaxError, MyiaDisconnectedCodeWarning
from ..utils import eprint
from ..ir import Graph

from .label import label
from .location import _show_location

def skip_node(node):
    """Whether to skip a step in the traceback based on ast node type."""
    return isinstance(node, (ast.If, ast.While, ast.For))


def _get_call(ref):
    ctx = ref.context
    g = ctx.graph
    while g and g.flags.get('auxiliary') \
            and ctx.parent and ctx.parent.graph:
        ctx = ctx.parent
        g = ctx.graph
    return g, ctx.argkey


def _get_loc(ref):
    node = ref.node
    if node.is_constant_graph():
        node = node.value
    return node.debug.find('location')


def _get_stack(error):
    refs = [*error.traceback_refs.values()]
    stack = []
    for ref in refs:
        if isinstance(ref, Reference):
            g, args = _get_call(ref)
            if g.flags.get('core'):
                continue
            loctype = 'direct'
            loc = _get_loc(ref)
        else:
            g, args = ref
            loctype = None
            loc = None
        if loc and skip_node(loc.node):
            continue
        stack.append((g, args, loctype, loc))
    return stack


class _PBlock:
    def __init__(self, title, separator, args, kwargs):
        self.title = title
        self.separator = separator
        self.args = args
        self.kwargs = kwargs


@pp.register_pretty(_PBlock)
def _pretty_pblock(pb, ctx):
    return pretty_struct(ctx, pb.title, pb.args, pb.kwargs)


@pp.register_pretty(data.PrimitiveFunction)
def _pretty_primfunc(x, ctx):
    return label(x.prim)


@pp.register_pretty(data.GraphFunction)
def _pretty_graphfunc(x, ctx):
    return label(x.graph)


def _format_call(fn, args):
    if isinstance(fn, Graph):
        kwargs = {label(p): arg for p, arg in zip(fn.parameters, args)}
        args = []
    else:
        kwargs = {}
    return format_abstract(_PBlock(label(fn), ' :: ', args, kwargs))


def print_inference_error(error):
    """Print an InferenceError's traceback."""
    stack = _get_stack(error)
    for fn, args, loctype, loc in stack:
        eprint('=' * 80)
        if loc is not None:
            eprint(f'{loc.filename}:{loc.line}')
        eprint('in', _format_call(fn, args))
        if loc is not None:
            _show_location(loc, '')
    eprint('~' * 80)
    eprint(f'{type(error).__name__}: {error.message}')


def print_myia_syntax_error(error):
    """Print MyiaSyntaxError's location."""
    loc = error.loc
    eprint('=' * 80)
    if loc is not None:
        eprint(f'{loc.filename}:{loc.line}')
    if loc is not None:
        _show_location(loc, '')
    eprint('~' * 80)
    eprint(f'{type(error).__name__}: {error}')


_previous_excepthook = sys.excepthook


def myia_excepthook(exc_type, exc_value, tb):
    """Print out InferenceError and MyiaSyntaxError specially."""
    if isinstance(exc_value, InferenceError):
        print_inference_error(exc_value)
    elif isinstance(exc_value, MyiaSyntaxError):
        print_myia_syntax_error(exc_value)
    else:
        _previous_excepthook(exc_type, exc_value, tb)


sys.excepthook = myia_excepthook


_previous_warning = warnings.showwarning


def myia_warning(message, category, filename, lineno, file, line):
    """Print out MyiaDisconnectedCodeWarning specially."""
    if category is MyiaDisconnectedCodeWarning:
        pass
    else:
        _previous_warning(message, category, filename, lineno, file, line)


warnings.showwarning = myia_warning
