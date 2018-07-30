"""Tools to print a traceback for an error in Myia."""

import ast
import sys

from ..dtype import Function
from ..infer import Inferrer, InferenceError
from ..prim import Primitive
from ..ir import is_apply, is_constant_graph

from .label import short_labeler as shl


def eprint(*things):
    """Print to stderr."""
    print(*things, file=sys.stderr)


def skip_node(node):
    """Whether to skip a step in the traceback based on ast node type."""
    return isinstance(node, (ast.If, ast.While, ast.For))


def _print_lines(lines, l1, c1, l2, c2, label='', mode='^'):
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

        if isinstance(mode, str):
            eprint(f'  {trimmed}')
            prefix = ' ' * (start + 2)
            eprint(prefix + '^' * (end - start) + label)
        else:
            prefix = trimmed[:start]
            hl = trimmed[start:end]
            rest = trimmed[end:]
            eprint(f'  {prefix}\033[1;31m{hl}\033[0m{rest}')


def _show_location(loc, ctx, label):
    eprint(f'File "{loc.filename}", line {loc.line}, column {loc.column}')
    eprint(ctx)
    with open(loc.filename, 'r') as contents:
        lines = contents.read().split('\n')
        _print_lines(lines, loc.line, loc.column,
                     loc.line_end, loc.column_end,
                     label, mode=5)


def _format_context(ctx):

    def fromsig(sig):
        sig = dict(sig)
        t = sig['type']
        shp = sig['shape']
        if shp:
            return f'{t}:{shp}'
        else:
            return t

    g = ctx.graph
    if g is None:
        return '<unknown>'
    args = [f'{shl.name(p)}: {fromsig(t)}'
            for (p, t) in zip(g.parameters, ctx.argkey)]
    return f'{shl.name(g)}({", ".join(args)})'


def _best_label(node, typ):
    if isinstance(typ, Inferrer):
        return _best_label(typ.identifier, None)
    elif isinstance(node, Primitive):
        return node.name
    else:
        return shl.name(node)


def _find_loc(ref):
    node = ref.node
    if is_constant_graph(node):
        node = node.value
    loc = node.debug.find('location')
    if loc is None:
        eprint('<Missing location for node>')
        return None
    return loc


async def _myia_multirefs(engine, error, refs):
    for i, ref in enumerate(refs):
        loc = _find_loc(ref)
        if loc is None:
            continue
        ctx = ref.context
        g = ctx.graph
        while g and g.flags.get('auxiliary') \
                and ctx.parent and ctx.parent.graph:
            ctx = ctx.parent
            g = ctx.graph
        ctx_str = 'in ' + _format_context(ctx)
        if g and g.flags.get('core', False):
            eprint(ctx_str)
            eprint('  Invalid signature for core function.')
            eprint()
            break
        _show_location(loc, ctx_str, str())
        eprint()

    eprint(f'{type(error).__qualname__}: {error.message}')


async def _myia_traceback(engine, error):
    refs = [*error.traceback_refs.values()]
    refs.reverse()

    if not refs:
        return await _myia_multirefs(engine, error, error.refs)

    ref = None
    for i, ref in enumerate(refs):
        loc = _find_loc(ref)
        if loc is None:
            continue
        if skip_node(loc.node):
            continue
        ctx = ref.context
        g = ctx.graph
        while g.flags.get('auxiliary') and ctx.parent and ctx.parent.graph:
            ctx = ctx.parent
            g = ctx.graph
        ctx_str = 'in ' + _format_context(ctx)
        if g.flags.get('core', False):
            eprint(ctx_str)
            eprint('  Invalid signature for core function.')
            eprint()
            break
        _show_location(loc, ctx_str, str())
        eprint()
    else:
        if ref is not None and is_apply(ref.node):
            ctx = ref.context
            irefs = [engine.ref(node, ctx) for node in ref.node.inputs]
            fn_ref, *_ = irefs
            fn_type, *arg_types = [await iref['type'] for iref in irefs]
            fn_str = _best_label(fn_ref.node, fn_type) or '<function>'
            if isinstance(fn_type, (Function, Inferrer)):
                types_str = ", ".join(map(str, arg_types))
                eprint(f'in {fn_str}({types_str})')
                eprint(f'  Invalid signature for core function.')
            else:
                eprint(f'Invalid function type:\n  {fn_str}: {fn_type}')
            eprint()

    eprint(f'{type(error).__qualname__}: {error.message}')


_previous_excepthook = sys.excepthook


def print_inference_error(err):
    """Print an InferenceError's traceback."""
    eng = err.engine
    eng.run_coroutine(_myia_traceback(eng, err), throw=True)


def myia_excepthook(exc_type, exc_value, tb):
    """Print out InferenceError specially."""
    if isinstance(exc_value, InferenceError):
        print_inference_error(exc_value)
    else:
        _previous_excepthook(exc_type, exc_value, tb)


sys.excepthook = myia_excepthook
