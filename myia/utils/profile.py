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
        self.ctx = self
        self.d = dict()

    def main(self):
        self.ctx = ProfContext('__total__', self)
        return self.ctx

    def step(self, name):
        self.ctx = ProfContext(name, self)
        return self.ctx

    def lap(self, count):
        self.ctx = ProfContext('Cycle {count}', self)
        return self.ctx

    def pop(self):
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
        if len(self.d) > 1:
            self.parent.d[self.name] = self.d
        else:
            self.parent.d[self.name] = self.d['__total__']
        self.p.pop()
        return None


class NoProf:
    def main(self):
        return DummyContext()

    def step(self, name):
        return DummyContext()


class NoProfContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return None


no_prof = NoProf()
