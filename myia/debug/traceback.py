"""Tools to print a traceback for an error in Myia."""

import re
import ast
import sys
from colorama import Fore, Style
import prettyprinter as pp

from .. import dtype
from ..abstract import InferenceError, Inferrer, Reference, data, ANYTHING, \
    TYPE, SHAPE, VALUE
from ..utils import eprint, overload
from ..ir import Graph

from .label import label


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


def get_stack(error):
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


def _pretty(ctx, title, args, kwargs, sep=' :: '):
    kwargs = {f'{k}<<{sep}>>': v
              for k, v in kwargs.items()}
    return pp.pretty_call_alt(ctx, str(title), args, kwargs)


@pp.register_pretty(_PBlock)
def pretty_pblock(pb, ctx):
    return _pretty(ctx, pb.title, pb.args, pb.kwargs)


@pp.register_pretty(data.AbstractClass)
def pretty_aclass(a, ctx):
    return _pretty(ctx, a.tag, [], a.attributes)


@pp.register_pretty(data.AbstractTuple)
def pretty_atuple(a, ctx):
    return pp.pretty_call_alt(ctx, "", a.elements, {})


@pp.register_pretty(data.AbstractScalar)
def pretty_atuple(a, ctx):
    t = a.values[TYPE]
    if dtype.ismyiatype(t, dtype.Float):
        rval = f'f{t.bits}'
    elif dtype.ismyiatype(t, dtype.Int):
        rval = f'i{t.bits}'
    elif dtype.ismyiatype(t, dtype.UInt):
        rval = f'u{t.bits}'
    else:
        rval = str(t)
    v = a.values[VALUE]
    if v is not ANYTHING:
        rval += f' = {v}'
    return rval


@pp.register_pretty(data.AbstractArray)
def pretty_aarray(a, ctx):
    elem = pp.pformat(a.element)
    shp = ' x '.join('?' if s is ANYTHING else str(s)
                   for s in a.values[SHAPE])
    return f'{elem} x {shp}'


@pp.register_pretty(data.AbstractFunction)
def pretty_afunc(a, ctx):
    return '|'.join([pp.pformat(fn) for fn in a.get_sync()])


@pp.register_pretty(data.PrimitiveFunction)
def pretty_primfunc(x, ctx):
    return label(x.prim)


@pp.register_pretty(data.GraphFunction)
def pretty_primfunc(x, ctx):
    return label(x.graph)


def _format_call(fn, args):
    if isinstance(fn, Graph):
        kwargs = {label(p): arg for p, arg in zip(fn.parameters, args)}
        args = []
    else:
        kwargs = {}
    rval = pp.pformat(_PBlock(label(fn), ' :: ', args, kwargs))
    rval = re.sub(r'<<([^>]+)>>=', r'\1', rval)
    return rval


def _show_location(loc, label):
    with open(loc.filename, 'r') as contents:
        lines = contents.read().split('\n')
        _print_lines(lines, loc.line, loc.column,
                     loc.line_end, loc.column_end,
                     label)


def _print_lines(lines, l1, c1, l2, c2, label='', mode='color'):
    for ln in range(l1, l2 + 1):
        line = lines[ln - 1]
        if ln == l1:
            trimmed = line.lstrip()
            to_trim = len(line) - len(trimmed)
            start = c1 - to_trim
        else:
            trimmed = line[to_trim:]
            start = 0

        if ln == l2:
            end = c2 - to_trim
        else:
            end = len(trimmed)

        if mode == 'color':
            prefix = trimmed[:start]
            hl = trimmed[start:end]
            rest = trimmed[end:]
            eprint(f'{ln}: {prefix}{Fore.RED}{Style.BRIGHT}'
                   f'{hl}{Style.RESET_ALL}{rest}')
        else:
            eprint(f'{ln}: {trimmed}')
            prefix = ' ' * (start + 2 + len(str(ln)))
            eprint(prefix + '^' * (end - start) + label)


def print_inference_error(error):
    """Print an InferenceError's traceback."""
    stack = get_stack(error)
    for fn, args, loctype, loc in stack:
        eprint('=' * 80)
        if loc is not None:
            eprint(f'{loc.filename}:{loc.line}')
        eprint('in', _format_call(fn, args))
        if loc is not None:
            _show_location(loc, '')
    eprint('~' * 80)
    eprint(f'{type(error).__name__}: {error.message}')


_previous_excepthook = sys.excepthook


def myia_excepthook(exc_type, exc_value, tb):
    """Print out InferenceError specially."""
    if isinstance(exc_value, InferenceError):
        print_inference_error(exc_value)
    else:
        _previous_excepthook(exc_type, exc_value, tb)


sys.excepthook = myia_excepthook
