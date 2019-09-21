"""Tracing mechanism for Myia, for debugging and profiling."""

import re
from collections import defaultdict
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass
from functools import wraps
from time import perf_counter


def glob_to_regex(glob):
    """Transforms a glob-like expression into a regular expression.

    * `**` matches any character sequence including the / delimiter
        * `/**/` can also match `/`
    * `*` matches any character sequence except /
    * If glob does not start with /, `/**/` is prepended
    """
    def replacer(m):
        if m.group() == '/**/':
            return r'(/.*/|/)'
        else:
            return r'[^/]*'

    if glob.startswith('**'):
        glob = f'/{glob}'
    elif not glob.startswith('/'):
        glob = f'/**/{glob}'
    if glob.endswith('**'):
        glob += '/*'

    patt = r'/\*\*/|\*'
    glob = re.sub(patt, replacer, glob)
    return re.compile(glob)


class Tracer:
    """Event-based tracer."""

    def __init__(self):
        """Initialize the Tracer."""
        self.stack = []
        self.curpath = ''
        self.listeners = []

    def emit(self, name, **kwargs):
        """Emit an event."""
        curpath = self.curpath + f'/{name}'
        for path, fn in self.listeners:
            if path.fullmatch(curpath):
                fn(**kwargs,
                   _event=name,
                   _stack=self.stack,
                   _curpath=curpath)

    def on(self, pattern, fn):
        """Register a function to trigger on a certain pattern.

        The pattern can be an event name or a glob.

        * `**` in a pattern matches any character sequence including
          the / delimiter
            * `/**/` can also match `/`
        * `*` matches any character sequence except /
        * A pattern that does not start with `/` is equivalent to the same
          pattern prepended with `/**/`, e.g. `apple` is equivalent to
          `/**/apple`
        """
        if not isinstance(pattern, re.Pattern):
            pattern = glob_to_regex(pattern)
        self.listeners.append((pattern, fn))

    def __copy__(self):
        cp = Tracer()
        cp.stack = list(self.stack)
        cp.curpath = self.curpath
        cp.listener = list(self.listeners)
        return cp

    def __call__(self, name, *args, **kwargs):
        """Start an enter/exit block using the given name."""
        return TracerContextManager(self, name, args, kwargs)

    def __getattr__(self, attr):
        if attr.startswith('emit_'):
            attr = attr[5:]
            return lambda **kwargs: self.emit(attr, **kwargs)
        elif attr.startswith('on_'):
            attr = attr[3:]
            return lambda fn: self.on(attr, fn)
        else:  # pragma: no cover
            return getattr(super(), attr)


class TracerContextManager:
    """Represents a tracing block that is entered and then exited."""

    def __init__(self, tracer, name, args, kwargs):
        """Initialize a TracerContextManager."""
        self.tr = tracer
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.results = {}

    def set_results(self, **results):
        """Set the block's results, which will be sent with the exit event."""
        self.results = results

    def __enter__(self):
        self.tr.stack.append(self)
        self.tr.curpath += f'/{self.name}'
        self.tr.emit('enter', _context=self, **self.kwargs)
        return self

    def __exit__(self, *_):
        self.tr.emit('exit', _context=self, **self.results)
        self.tr.stack.pop()
        self.tr.curpath = ''.join(f'/{x.name}' for x in self.tr.stack)

    def __str__(self):
        return f'<TracerContextManager {self.name}>'

    __repr__ = __str__


_tracer = ContextVar('tracer', default=Tracer())


def tracer(name=None, **kwargs):
    """Return or use the current tracer.

    Returns:
        * With no arguments, returns the current tracer.
        * With arguments, returns a TracerContextManager that may be used
          with the `with` statement

    """
    v = _tracer.get()
    if name is not None:
        return v(name, **kwargs)
    else:
        assert not kwargs
        return v


class TraceListener:
    """Represents a collection of listeners on a tracer.

    Arguments:
        focus: A glob prepended to this listener's patterns.
    """

    def __init__(self, focus=None):
        """Initialize a TraceListener."""
        self.focus = focus

    def install(self, tracer):
        """Install the listeners on the tracer."""
        for method_name in dir(self):
            if method_name.startswith('on_'):
                ev = method_name[3:]
                patt = f'{self.focus}/{ev}' if self.focus else ev
                tracer.on(patt, getattr(self, method_name))

    def post(self):
        """Do things after the process is completed."""
        pass

    def __enter__(self):
        self.tracer = copy(_tracer.get())
        self.token = _tracer.set(self.tracer)
        self.install(self.tracer)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        _tracer.reset(self.token)
        self.post()


class TraceExplorer(TraceListener):
    """Print out all distinct events and their arguments."""

    def __init__(self, focus=None):
        """Initialize a TraceExplorer."""
        super().__init__(focus)
        self.paths = defaultdict(lambda: defaultdict(set))

    def install(self, tracer):
        """Install the TraceExplorer."""
        patt = self.focus or '**'
        tracer.on(patt, self._log_keys)

    def _log_keys(self, _curpath=None, **kwargs):
        d = self.paths[_curpath]
        for k, v in kwargs.items():
            if not k.startswith('_'):
                d[k].add(type(v))

    def post(self):
        """Print the events and their arguments."""
        for path, keys in self.paths.items():
            print(path)
            for key in sorted(keys):
                typenames = {v.__qualname__ for v in keys[key]}
                print('    ', key, '::', ' | '.join(typenames))


class DoTrace(TraceListener):
    """Register user patterns on the tracer.

    Arguments:
        patterns: A dict of glob or regex over event paths to a handler
            function.
        focus: A glob that will be prepended to all patterns to restrict
            them further, or None if the patterns are to be used as they
            are. This argument will not work if any pattern is a regex.
        post: A function to call at the end of processing. Defaults to
            None, which means no function is called.
    """

    def __init__(self, patterns={}, *, focus=None, post=None):
        """Initialize a DoTrace."""
        super().__init__(focus)
        self.patterns = patterns
        self._post = post

    def install(self, tracer):
        """Install the patterns."""
        for patt, fn in self.patterns.items():
            if self.focus:
                patt = self.focus + patt
            tracer.on(patt, fn)

    def post(self):
        """Execute the postprocessing function, if there is one."""
        if self._post:
            self._post()


def _unit(secs):
    ms = secs * 1000
    return f"{ms:10.2f}ms"


@dataclass
class ProfileResults(dict):
    """Contains profiling results for a tracing block."""

    name: str
    start = None
    end = None
    total = None
    parts_total = None
    overhead = None

    def __init__(self, name=None):
        """Initialize a ProfileResults."""
        self.name = name

    def print(self, *, indent=0):
        """Print a visualisation of a profile."""
        ind = " " * indent

        if self.name is not None and self.total is not None:
            print(f'{ind}{self.name:30}{_unit(self.total)}')
            indent += 3
            ind = " " * indent

        if self.overhead:
            print(f'{ind}{"[overhead]":30}{_unit(self.overhead)}')
        for prof2 in self.values():
            prof2.print(indent=indent)


class Profiler(TraceListener):
    """Build a profile of the execution of the program."""

    def __init__(self, focus=None, print_results=True):
        """Initialize a Profiler."""
        super().__init__(focus)
        self.hierarchical = ProfileResults()
        self.aggregate = defaultdict(list)
        self.overhead = 0
        self.print_results = print_results

    def on_enter(self, _stack=None, **kwargs):
        """Executed when a block is entered."""
        d = self.hierarchical
        for part in _stack:
            d.setdefault(part.name, ProfileResults(part.name))
            d = d[part.name]
        d.start = perf_counter()

    def on_exit(self, _stack=None, **kwargs):
        """Executed when a block is exited."""
        d = self.hierarchical
        for part in _stack:
            d = d[part.name]
        d.end = perf_counter()
        d.total = d.end - d.start
        d.parts_total = sum(v.total for v in d.values())
        if d.parts_total:
            d.overhead = d.total - d.parts_total
        else:
            d.overhead = 0
        self.overhead += d.overhead
        self.aggregate[part.name].append(d)

    def post(self):
        """Print the results."""
        if self.print_results:
            print('====================')
            print('Hierarchical profile')
            print('====================')
            self.hierarchical.print()
            print()
            print('==========')
            print('Summations')
            print('==========')
            if self.overhead:
                print(f'{"[overhead]":20}{_unit(self.overhead)}')
            for k, v in self.aggregate.items():
                if len(v) > 1:
                    tot = sum(x.total for x in v)
                    print(f'{k:20}{_unit(tot)}')


def listener(*patterns):
    """Create a listener for one or more patterns on a tracer."""
    def deco(fn):
        @wraps(fn)
        def new_fn(**kwargs):
            return DoTrace({pattern: fn for pattern in patterns}, **kwargs)
        return new_fn
    return deco


__consolidate__ = True
__all__ = [
    'DoTrace',
    'ProfileResults',
    'Profiler',
    'TraceExplorer',
    'TraceListener',
    'Tracer',
    'TracerContextManager',
    'listener',
    'tracer',
]
