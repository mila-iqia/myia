"""Tools to print a traceback for an error in Myia."""

import ast
import sys
import warnings

import prettyprinter as pp
from colorama import Fore, Style

from ..abstract import Reference, data, format_abstract, pretty_struct
from ..ir import ANFNode, Graph
from ..parser import Location, MyiaDisconnectedCodeWarning, MyiaSyntaxError
from ..utils import InferenceError
from .label import label


def skip_node(node):
    """Whether to skip a step in the traceback based on ast node type."""
    return isinstance(node, (ast.If, ast.While, ast.For))


def _get_call(ref):
    ctx = ref.context
    g = ctx.graph or ref.node.graph
    while g and g.has_flags('auxiliary') \
            and ctx.parent and ctx.parent.graph:
        ctx = ctx.parent
        g = ctx.graph
    return g, ctx.argkey


def _get_loc(node):
    if node.is_constant_graph():
        node = node.value
    loc = node.debug.find('location')
    genfn = None
    if loc is None:
        tr = node.debug.find('trace', skip={'copy', 'equiv'})
        if tr:
            idx = len(tr) - 3
            while idx >= 0:
                fr = tr[idx]
                if 'myia/myia/ir' in fr.filename:
                    idx -= 1
                    continue
                loc = Location(
                    fr.filename, fr.lineno, 0, fr.lineno, 0, None
                )
                genfn = fr.name
                break
    return loc, genfn


def _get_stack(error):
    refs = [*error.traceback_refs.values()] + error.refs
    stack = []
    for ref in refs:
        if isinstance(ref, Reference):
            g, args = _get_call(ref)
            if g.has_flags('core'):
                continue
            loctype = 'direct'
            loc, genfn = _get_loc(ref.node)
        elif isinstance(ref, ANFNode):
            g = ref.graph
            args = None
            loctype = 'direct'
            loc, genfn = _get_loc(ref)
        else:
            g, args = ref
            loctype = None
            loc = None
            genfn = None
        if loc and skip_node(loc.node):
            continue
        stack.append((g, args, loctype, loc, genfn))
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
    if args is None:
        return label(fn)
    if isinstance(fn, Graph):
        kwargs = {label(p): arg for p, arg in zip(fn.parameters, args)}
        args = []
    else:
        kwargs = {}
    return format_abstract(_PBlock(label(fn), ' :: ', args, kwargs))


def _show_location(loc, label, mode=None, color='RED', file=sys.stderr):
    with open(loc.filename, 'r') as contents:
        lines = contents.read().split('\n')
        _print_lines(lines, loc.line, loc.column,
                     loc.line_end, loc.column_end,
                     label, mode, color, file=file)


def _print_lines(lines, l1, c1, l2, c2, label='',
                 mode=None, color='RED', file=sys.stderr):
    if mode is None:
        if file.isatty():
            mode = 'color'
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
            print(f'{ln}: {prefix}{getattr(Fore, color)}{Style.BRIGHT}'
                  f'{hl}{Style.RESET_ALL}{rest}',
                  file=file)
        else:
            print(f'{ln}: {trimmed}', file=file)
            prefix = ' ' * (start + 2 + len(str(ln)))
            print(prefix + '^' * (end - start) + label, file=file)


def print_inference_error(error, file=sys.stderr):
    """Print an InferenceError's traceback."""
    stack = _get_stack(error)
    for fn, args, loctype, loc, genfn in stack:
        print('=' * 80, file=file)
        if loc is not None:
            print(f'{loc.filename}:{loc.line}', file=file)
        gen = f'via code generated in {genfn}:' if genfn else ''
        print('in', _format_call(fn, args), gen, file=file)
        if loc is not None:
            _show_location(loc, '', file=file)
    print('~' * 80, file=file)
    if error.pytb:
        print(error.pytb, file=file)
    else:
        print(f'{type(error).__name__}: {error.message}', file=file)


def print_myia_syntax_error(error, file=sys.stderr):
    """Print MyiaSyntaxError's location."""
    loc = error.loc
    print('=' * 80, file=file)
    if loc is not None:
        print(f'{loc.filename}:{loc.line}', file=file)
    if loc is not None:
        _show_location(loc, '', file=file)
    print('~' * 80, file=file)
    print(f'{type(error).__name__}: {error}', file=file)


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


def print_myia_warning(warning, file=sys.stderr):
    """Print Myia Warning's location."""
    msg = warning.args[0]
    loc = warning.loc
    print('=' * 80, file=file)
    if loc is not None:
        print(f'{loc.filename}:{loc.line}', file=file)
    if loc is not None:
        _show_location(loc, '', None, 'MAGENTA', file=file)
    print('~' * 80, file=file)
    print(f'{warning.__class__.__name__}: {msg}', file=file)


_previous_warning = warnings.showwarning


def myia_warning(message, category, filename, lineno, file, line):
    """Print out MyiaDisconnectedCodeWarning specially."""
    if category is MyiaDisconnectedCodeWarning:
        # message is actually a MyiaDisconnectedCodeWarning object,
        # even though this parameter of myia_warning is called message
        # (in order to match parameter names of overrided showwarning)
        print_myia_warning(message)
    else:
        _previous_warning(message, category, filename, lineno, file, line)


warnings.showwarning = myia_warning
warnings.filterwarnings('always', category=MyiaDisconnectedCodeWarning)
