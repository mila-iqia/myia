"""Utility injecting functionality, only use for debugging."""

import sys
import pdb
from .buche import buche
from buche.debug import BucheDb
from buche import reader


def bubrk():
    """Breakpoint using buche."""
    print()
    ch = buche.open('debug', 'log', dict(hasInput=True))
    db = BucheDb(ch, reader)
    db.set_trace()


suite = {
    'buche': buche,
    'pdb': pdb,
    'breakpoint': pdb.set_trace,
    'bubrk': bubrk
}


def inject(**utilities):
    """Inject all utilities in the globals of every module."""
    for name, module in list(sys.modules.items()):
        glob = vars(module)
        for key, value in utilities.items():
            if key not in glob:
                try:
                    glob[key] = value
                except TypeError as e:
                    pass


def inject_suite():
    """Inject default utilities in the globals of every module."""
    inject(**suite)
