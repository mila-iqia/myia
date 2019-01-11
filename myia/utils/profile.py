"""Utilities to help support profiling."""

from time import perf_counter as prof_counter  # noqa


def _unit(secs):  # pragma: no cover
    return "%.3gs" % secs


def print_profile(prof, *, indent=0):  # pragma: no cover
    """Print a visualisation of a profile."""
    total = prof.pop('__total__')
    runtime = 0
    ind = " " * indent

    for v in prof.values():
        if isinstance(v, dict):
            v = v['__total__']
        runtime += v

    overhead = total - runtime

    print(f"{ind}Total time taken: {_unit(total)}")
    print(f"{ind} Overhead: {_unit(overhead)}")
    print(f"{ind} Time spent in runtime: {_unit(runtime)}")
    for k, v in prof.items():
        if isinstance(v, dict):
            print(f"{ind}  {k}:")
            print_profile(v, indent=indent + 4)
        else:
            print(f"{ind}  {k}: {_unit(v)}")


class Profile:
    def __init__(self):
        self.ctx = None
        self.ctx = ProfContext(None, self)
        self.d = dict()
        self.ctx.d = self.d

    def __enter__(self):
        self.ctx.start = prof_counter()
        return self.ctx

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.ctx.stop = prof_counter()
        self.ctx.d['__total__'] = self.ctx.stop - self.ctx.start
        return None

    def step(self, name):
        self.ctx = ProfContext(name, self)
        return self.ctx

    def lap(self, count):
        self.ctx = ProfContext(f'Cycle {count}', self)
        return self.ctx

    def _pop(self):
        assert self.ctx.name is not None
        self.ctx = self.ctx.parent


class ProfContext:
    def __init__(self, name, p):
        self.name = name
        self.p = p
        self.parent = p.ctx
        self.d = dict()

    def __enter__(self):
        self.start = prof_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = prof_counter()
        self.d['__total__'] = self.stop - self.start
        if self.parent:
            if len(self.d) > 1:
                self.parent.d[self.name] = self.d
            else:
                self.parent.d[self.name] = self.d['__total__']
            self.p._pop()
        return None


class NoProf:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return None

    def step(self, name):
        return NoProfContext()

    def lap(self, count):
        return NoProfContext()


class NoProfContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return None


no_prof = NoProf()
