"""Utilities to debug Myia.

Using pytest:

    pytest -s -T debug.trace.<name>:<arg1>:<arg2>... ...
    pytest -s -T 'debug.trace.<name>(<arg1>,<arg2>)' ...

For example:

    -T debug.trace.prof         # profile
    -T debug.trace.explore      # look at available events and fields

    -T debug.trace.log          # list events in order
    -T debug.trace.log:opt      # list all opt events
    -T debug.trace.log:opt:opt  # show the opt field for all opt events
    -T debug.trace.log:+opt:opt # show the opts from step_opt

    # compare node count before and after a successful opt
    -T debug.trace.compare:+opt:+optname:+countnodes

    -T debug.trace.graph        # show final graph (requires Buche)

The '+' prefix taps pre-made paths or rules:

* +xyz, as a path argument, refers to debug.trace._path_xyz
* +xyz, as a field, refers to debug.trace._rule_xyz

"""

import os
import time
from collections import Counter, defaultdict
from decimal import Decimal

import breakword
from colorama import Fore

from myia.utils import (  # noqa: F401
    DoTrace,
    Profiler as prof,
    TraceExplorer as explore,
    TraceListener,
)

from .inject import bucheg

_beginning = time.monotonic()
_current = time.monotonic()


#############
# Utilities #
#############


class Time:
    def __init__(self, t=None):
        if t is None:
            self.t = time.monotonic() - _beginning
        else:
            self.t = t

    def compare(self, other):
        return Time(other.t - self.t)

    @classmethod
    def statistics(cls, tdata):
        data = [t.t for t in tdata]
        print(f"  Min:", Time(min(data)))
        print(f"  Avg:", Time(sum(data) / len(data)))
        print(f"  Max:", Time(max(data)))

    def __str__(self):
        d = Decimal(self.t)
        unit = "s"
        units = ["ms", "us", "ns"]
        for other_unit in units:
            if d >= 1:
                break
            else:
                d *= 1000
                unit = other_unit
        d = round(d, 3)
        return f"{d}{unit}"


def _color(color, text):
    """Wrap the text with the given color.

    If Buche is active, the color is not applied.
    """
    if os.environ.get("BUCHE"):
        return text
    else:
        return f"{color}{text}{Fore.RESET}"


def _pgraph(path):
    """Print a graph using Buche."""

    def _p(graph, **_):
        bucheg(graph)

    return lambda: DoTrace({path: _p})


class Getters(dict):
    def __init__(self, fields, kwfields):
        for field in fields:
            if field == "help":
                self[field] = lambda **kwargs: ", ".join(kwargs)
            elif field.startswith("+"):
                field = field[1:]
                self[field] = globals()[f"_rule_{field}"]
            else:
                self[field] = self._get_by_name(field)
        for name, getter in kwfields.items():
            self[name] = getter

    def _get_by_name(self, field):
        def _get(**kwargs):
            return kwargs.get(field, f"<{field} NOT FOUND>")

        return _get

    def __call__(self, kwargs):
        results = {name: getter(**kwargs) for name, getter in self.items()}
        return results


def _display(curpath, results, word=None, brk=True):
    w = word or breakword.word()
    if len(results) == 0:
        print(w, curpath)
    elif len(results) == 1:
        _, value = list(results.items())[0]
        print(w, _color(Fore.LIGHTBLACK_EX, curpath), value)
    else:
        print(w, _color(Fore.LIGHTBLACK_EX, curpath))
        for name, value in results.items():
            print(f"  {name}: {value}")
    if brk:
        _brk(w)


def _brk(w):
    if breakword.after():
        print("Breaking on:", w)
        breakpoint(skip=["debug.*", "myia.utils.trace"])


def _resolve_path(p, variant=""):
    if not p:
        rval = "**"
    elif p.startswith("+"):
        rval = globals()[f"_path{variant}_{p[1:]}"]
    else:
        rval = p
    if isinstance(rval, str):
        rval = [rval]
    return rval


###########
# Tracers #
###########


# Print the final graph
graph = _pgraph("step_validate/enter")


# Print the graph after monomorphization
graph_mono = _pgraph("step_specialize/exit")


# Print the graph after parsing
graph_parse = _pgraph("step_parse/exit")


def log(path=None, *fields, **kwfields):
    """Log fields of interest on the given path.

    The breakword module is used for logging, thus it is possible to set a
    word upon which to enter a breakpoint (using the BREAKWORD environment
    variable).

    * When no path is given, show all events.
    * The "help" field shows all possible fields.
    """

    getters = Getters(fields, kwfields)

    def _p(**kwargs):
        _curpath = kwargs["_curpath"]
        results = getters(kwargs)
        _display(_curpath, results)

    return DoTrace({pth: _p for pth in _resolve_path(path)})


def opts():
    """Log the optimizations applied during the opt phase."""
    return log("step_opt/**/opt/success", opt=lambda opt, **_: opt.name)


def compare(path=None, *fields, **kwfields):
    store = {}
    getters = Getters(fields, kwfields)

    def _compare(old, new):
        if isinstance(old, dict):
            return {k: _compare(v, new[k]) for k, v in old.items()}
        elif isinstance(old, (int, float)):
            diff = new - old
            if diff == 0:
                return old
            c = Fore.LIGHTGREEN_EX if diff > 0 else Fore.LIGHTRED_EX
            diff = f"+{diff}" if diff > 0 else str(diff)
            return f"{old} -> {new} ({_color(c, diff)})"
        elif hasattr(old, "compare"):
            return old.compare(new)
        elif old == new:
            return old
        else:
            return f"{old} -> {new}"

    def _enter(_curpath, **kwargs):
        _path = _curpath[:-6]
        w = breakword.word()
        store[_path] = (w, getters(kwargs))
        _brk(w)

    def _exit(_curpath, **kwargs):
        if "success" in kwargs and not kwargs["success"]:
            return
        _path = _curpath[:-5]
        w, old = store[_path]
        new = getters(kwargs)
        _display(_path, _compare(old, new), word=w, brk=False)

    path = _resolve_path(path, variant="cmp")
    return DoTrace({f"{path}/enter": _enter, f"{path}/exit": _exit,})


class StatAccumulator(TraceListener):
    def __init__(self, path, fields, kwfields):
        """Initialize a StatAccumulator."""
        self.path = _resolve_path(path)
        self.accum = defaultdict(list)
        self.getters = Getters(fields, kwfields)

    def install(self, tracer):
        """Install the StatAccumulator."""
        patt = self.path or "**"
        tracer.on(patt, self._do)

    def _do(self, **kwargs):
        for k, v in self.getters(kwargs).items():
            self.accum[(k, type(v))].append(v)

    def post(self):
        for (name, typ), data in self.accum.items():
            print(f"{name}:")
            if not data:
                print("  No data.")

            if issubclass(typ, (int, float)):
                print(f"  Min:", min(data))
                print(f"  Avg:", sum(data) / len(data))
                print(f"  Max:", max(data))

            elif hasattr(typ, "statistics"):
                typ.statistics(data)

            else:
                counts = Counter(data)
                align = max(len(str(obj)) for obj in counts)
                counts = sorted(counts.items(), key=lambda k: -k[1])
                for obj, count in counts:
                    print(f"  {str(obj).ljust(align)} -> {count}")


def stat(path=None, *fields, **kwfields):
    """Collect and display statistics about certain fields.

    * Numeric fields will display min/max/avg
    * String/other fields will count occurrences, sorted descending
    """
    return StatAccumulator(path, fields, kwfields)


#########
# Paths #
#########


_path_opt = ["step_opt/**/opt/success", "step_opt2/**/opt/success"]
_pathcmp_opt = ["step_opt/**/opt", "step_opt2/**/opt"]


#########
# Rules #
#########


def _rule_optname(opt=None, **kwargs):
    if opt is None:
        return "<NOT FOUND>"
    return opt.name


def _rule_optparam(node=None, **kwargs):
    if node is None:
        return "<NOT FOUND>"
    try:
        return str(node.inputs[1])
    except Exception:
        return "<???>"


def _rule_countnodes(graph=None, manager=None, **kwargs):
    if manager is None:
        if graph is None:
            return "<NOT FOUND>"
        if graph._manager is None:
            return "<NO MANAGER>"
        manager = graph.manager
    return len(manager.all_nodes)


def _rule_countgraphs(graph=None, manager=None, **kwargs):
    if manager is None:
        if graph is None:
            return "<NOT FOUND>"
        if graph._manager is None:
            return "<NO MANAGER>"
        manager = graph.manager
    return len(manager.graphs)


def _rule_time(**kwargs):
    return Time()


def _rule_reltime(**kwargs):
    global _current
    old = _current
    _current = time.monotonic()
    return Time(_current - old)
