
from typing import Tuple as TupleT
import inspect
import textwrap
import ast
from .parse import Parser, Locator, get_global_parse_env
from .stx import Symbol, ParseEnv, _Assign


def parse_source(url: str,
                 line: int,
                 src: str,
                 **kw) -> TupleT[Symbol, ParseEnv]:
    """
    Parse a source string with Myia.

    Arguments:
        url: The filename from whence the source comes.
        line: The line number at which the source starts.
        src: The source code to parse.
        kw: Keyword arguments passed to Parser.

    Returns:
        A pair:
        * The Symbol reference associated to the parsed
          function.
        * The ParseEnv that contains all the bindings that
          were created as a result of parsing. This includes
          the main function being parsed, but also auxiliary
          functions for loop bodies etc.
        To get the Lambda object that corresponds to the
        given source function, index the ParseEnv with the
        Symbol.
    """
    tree = ast.parse(src)
    p = Parser(Locator(url, line),
               get_global_parse_env(url),
               top_level=True,
               **kw)
    r = p.visit(tree, allow_decorator=True)
    if isinstance(r, list):
        r, = r
    if isinstance(r, _Assign):
        r = r.value
    genv = p.global_env
    assert genv is not None
    assert isinstance(r, Symbol)
    return r, genv


def parse_function(fn, **kw) -> TupleT[Symbol, ParseEnv]:
    """
    Parse a function with Myia.

    Arguments:
        fn: A Python function.
        kw: Keyword arguments passed to Parser.

    Returns:
        See ``parse_source``.
    """
    _, line = inspect.getsourcelines(fn)
    return parse_source(inspect.getfile(fn),
                        line,
                        textwrap.dedent(inspect.getsource(fn)),
                        **kw)


def make_error_function(data):
    def _f(*args, **kwargs):
        raise Exception(
            f"Function {data['name']} is for internal use only."
        )
    _f.data = data
    return _f


def myia(fn):
    # This is probably broken.
    _, genv = parse_function(fn)
    gbindings = genv.bindings
    glob = fn.__globals__
    bindings = {k: make_error_function({"name": k, "ast": v, "globals": glob})
                for k, v in gbindings.items()}
    glob.update(bindings)
    fsym = Symbol(fn.__name__, namespace='global')
    fn.data = bindings[fsym].data
    fn.associates = bindings
    return fn
