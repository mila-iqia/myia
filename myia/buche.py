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

Note: this functionality being somewhat generic, it could eventually
move to a standalone package.
"""


import os
import json
from hrepr import StdHRepr


_css_path = f'{os.path.dirname(__file__)}/myia.css'
_css = None


class HReprBase:
    @classmethod
    def __hrepr_resources__(cls, H):
        global _css
        if _css is None:
            _css = open(_css_path).read()
        return H.style(_css)


class HRepr(StdHRepr):
    def __default_handlers__(self):
        h = super().__default_handlers__()
        h.update({
            Exception: self.handle_Exception
        })
        return h

    def handle_Exception(self, obj, H, hrepr):
        return repr(obj)


class MasterBuche:
    def __init__(self, hrepr):
        self.hrepr = HRepr()
        self.resources = set()

    def raw(self, d={}, **params):
        message = {**d, **params}
        print(json.dumps(message), flush=True)

    def log(self, msg, **params):
        self.raw(command='log',
                 contents=str(msg),
                 **params)

    def show(self, obj, path='/', kind='log', **hrepr_params):
        r = self.hrepr(obj)  # , **hrepr_params)
        for res in self.hrepr.resources:
            if res not in self.resources:
                self.log(res,
                         path='/',
                         format='html')
                self.resources.add(res)
        self.log(r, format='html', kind=kind, path=path)


class Buche:
    def __init__(self, master, channel):
        self.master = master
        self.channel = channel
        self.to_open = {}

    def raw(self, path=None, **params):
        if path is None:
            path = self.channel
        else:
            path = self.join_path(path)
        self.master.raw(path=self.channel, **params)

    def pre(self, message):
        self.raw(command = 'log', format = 'pre', contents = message)

    def text(self, message):
        self.raw(command = 'log', format = 'text', contents = message)

    def html(self, message):
        self.raw(command = 'log', format = 'html', contents = message)

    def markdown(self, message):
        self.raw(command = 'log', format = 'markdown', contents = message)

    def open(self, name, type, force=False, **params):
        if force:
            subchannel = self.join_path(name)
            self.master.raw(command='open',
                            path=subchannel,
                            type=type,
                            **params)
        else:
            self.to_open[name] = (type, params)

    def join_path(self, p):
        return f'{self.channel.rstrip("/")}/{p.rstrip("/")}'

    def __getitem__(self, item):
        subchannel = self.join_path(item)
        info = self.to_open.get(item, None)
        if info:
            del self.to_open[item]
            type, params = info
            self.open(item, type, force=True, **params)
        return Buche(self.master, subchannel)

    def __call__(self, obj, kind="log", **params):
        self.master.show(obj, hrepr_params=params,
                         path=self.channel, kind=kind)


master = MasterBuche(StdHRepr())
buche = Buche(master, '/')
