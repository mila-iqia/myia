"""Tools to print a traceback for an error in Myia."""

import ast
import sys
from colorama import Fore, Style

from ..abstract import InferenceError
from ..utils import eprint

from .label import label


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
            eprint(f'  {prefix}{Fore.RED}{Style.BRIGHT}'
                   f'{hl}{Style.RESET_ALL}{rest}')


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
    args = [f'{label(p)}: {fromsig(t)}'
            for (p, t) in zip(g.parameters, ctx.argkey)]
    return label(g), f'({", ".join(args)})'


def _label(node, typ):
    if isinstance(typ, Inferrer):
        return _label(typ.identifier, None)
    else:
        return label(node)


def _find_loc(ref):
    node = ref.node
    if node.is_constant_graph():
        node = node.value
    loc = node.debug.find('location')
    if loc is None:
        eprint('<Missing location for node>')
        return None
    return loc


def _get_main(ref):
    ctx = ref.context
    g = ctx.graph
    while g and g.flags.get('auxiliary') \
            and ctx.parent and ctx.parent.graph:
        ctx = ctx.parent
        g = ctx.graph
    return g, ctx


async def _myia_traceback(engine, error):
    refs = [*error.traceback_refs.values()]
    ref = None
    for i, ref in enumerate(refs):
        loc = _find_loc(ref)
        if loc is None:
            continue
        if skip_node(loc.node):
            continue
        g, ctx = _get_main(ref)
        fstr, argstr = _format_context(ctx)
        if g.flags.get('core', False):
            error.print_tb_end(fstr, argstr, False)
            break
        ctx_str = f'in {fstr}{argstr}'
        _show_location(loc, ctx_str, str())
        eprint()
    else:
        fstr, argstr = None, None
        if ref is not None and ref.node.is_apply():
            ctx = ref.context
            irefs = [engine.ref(node, ctx) for node in ref.node.inputs]
            fn_ref, *_ = irefs
            try:
                fn_type, *arg_types = [await iref['type'] for iref in irefs]
                fstr = _label(fn_ref.node, fn_type) or '<function>'
                argstr = "(" + ", ".join(map(str, arg_types)) + ")"
            except InferenceError:
                fstr = None
                argstr = None
        error.print_tb_end(fstr, argstr, True)


_previous_excepthook = sys.excepthook


def print_inference_error(err):
    """Print an InferenceError's traceback."""
    eng = err.engine
    eng.run_coroutine(_myia_traceback(eng, err), throw=True)


def myia_excepthook(exc_type, exc_value, tb):
    """Print out InferenceError specially."""
    if isinstance(exc_value, InferenceError) and hasattr(exc_value, 'engine'):
        print_inference_error(exc_value)
    else:
        _previous_excepthook(exc_type, exc_value, tb)


sys.excepthook = myia_excepthook
