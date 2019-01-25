"""Utilities to help support profiling."""

from time import perf_counter as prof_counter  # noqa


def _unit(secs):
    return "%.3gs" % secs


def print_profile(prof, *, indent=0):
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
    """Class to collect a hierachical profile of activities.

    Every profile is in a context (except the top-level one) and has a
    name associated with it in its parent context.

    It is expected that the sum of sub-profiles will equal the time
    spent in a specific step.  Any difference is reported as overhead.

    A profile is delimited using the python with statement:

        with profile:
            # do something

    For sub-profiles, use one of the appropriate methods and next the
    with statements.  The nesting can be through functions calls of
    other forms of control flow.

    """

    def __init__(self):
        """Create a Profile with its initial context."""
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

    def print(self):
        """Print a formatted version of the profile."""
        return print_profile(self.d)

    def step(self, name):
        """Start a step in the current context with the given name.

        Nomes must be unique otherwise the previous record will be
        overwritten.

            with profile:
                with profile.step('start'):
                    # Start stuff
                with profile.step('end'):
                    # End stuff

        """
        self.ctx = ProfContext(name, self)
        return self.ctx

    def lap(self, count):
        """Creates subcontext for a repeated action.

        Count should be monotonically increasing.

            with profile:
                for i in range(10):
                    with profile.lap(i):
                        # loop stuff

        """
        self.ctx = ProfContext(f'Cycle {count}', self)
        return self.ctx

    def _pop(self):
        assert self.ctx.name is not None
        self.ctx = self.ctx.parent


class ProfContext:
    """Utility class for Profile."""

    def __init__(self, name, p):
        """Initialize a subcontext."""
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
    """Class that mimics Profile, but does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return None

    def print(self):
        """Does nothing."""
        pass

    def step(self, name):
        """Does nothing."""
        return self

    def lap(self, count):
        """Does nothing."""
        return self


no_prof = NoProf()
