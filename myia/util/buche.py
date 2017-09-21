"""
Helpers to use the buche logger (https://github.com/breuleux/buche)
which is used by ``python -m myia inspect``

Buche can log HTML views of objects in multiple tabs, which is useful
to inspect or debug the monstrous contraptions Grad generates.

Usage:

    from .buche import buche

    buche['mylog'](some_object)  # log in /mylog
    buche.open('otherlog')
    other = buche['otherlog']
    other.markdown('**TODO**: write HTML.')
    other.html('<s><b>DONE<b><s>: write HTML.')

Then, to run it:

    $ buche -c 'python -u script.py'

Or:

    $ python -u script.py | buche

The ``-u`` flag is not, strictly speaking, necessary, but it forces
Python to flush its output on every print, so it'll look more
responsive.
"""

import os
from buche import *


_css_path = f'{os.path.dirname(__file__)}/myia.css'
_css = None


class HReprBase:
    @classmethod
    def __hrepr_resources__(cls, H):
        global _css
        if _css is None:
            _css = open(_css_path).read()
        return H.style(_css)
