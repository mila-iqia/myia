"""Utilities to help support profiling."""

from time import perf_counter as prof_counter  # noqa


def _unit(secs):
    return "%.3gs" % secs


def print_profile(prof, *, indent=0):
    """Print a visualisation of a profile."""
    total = prof.pop('__total__')
    count = prof.pop('__count__', None)
    runtime = 0
    ind = " " * indent

    for v in prof.values():
        if isinstance(v, dict):
            v = v['__total__']
        runtime += v

    overhead = total - runtime

    print(f"{ind}Total time taken: {_unit(total)}")
    if count is not None:
        print(f"{ind} Number of loops: {count}")
    print(f"{ind} Overhead: {_unit(overhead)}")
    print(f"{ind} Time spent in runtime: {_unit(runtime)}")
    for k, v in prof.items():
        if isinstance(v, dict):
            print(f"{ind}  {k}:")
            print_profile(v, indent=indent + 4)
        else:
            print(f"{ind}  {k}: {_unit(v)}")
