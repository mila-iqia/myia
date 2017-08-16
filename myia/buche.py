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

from typing import Any

import os
import sys
import json
import weakref
import traceback
from hrepr import StdHRepr
from .util import Props
from .event import EventDispatcher


_css_path = f'{os.path.dirname(__file__)}/myia.css'
_css = None


id_registry: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


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

    def __call__(self, obj, **kwargs):
        res = super().__call__(obj, **kwargs)
        if 'obj-id' in res.attributes:
            the_id = int(res.attributes['obj-id'])
            id_registry[the_id] = obj
        return res

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
                self.raw(
                    command = 'resource',
                    path = '/',
                    type = 'direct',
                    contents = str(res)
                )
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
        elif path.startswith('/'):
            pass
        else:
            path = self.join_path(path)
        self.master.raw(path=path, **params)

    def pre(self, message, **kwargs):
        self.raw(command = 'log', format = 'pre',
                 contents = message, **kwargs)

    def text(self, message, **kwargs):
        self.raw(command = 'log', format = 'text',
                 contents = message, **kwargs)

    def html(self, message, **kwargs):
        self.raw(command = 'log', format = 'html',
                 contents = message, **kwargs)

    def markdown(self, message, **kwargs):
        self.raw(command = 'log', format = 'markdown',
                 contents = message, **kwargs)

    def open(self, name, type, force=False, **params):
        if force:
            subchannel = self.join_path(name)
            self.master.raw(command='open',
                            path=subchannel,
                            type=type,
                            **params)
            return self[name]
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


def handle_exception(e, H, hrepr):
    tb = e.__traceback__
    entries = traceback.extract_tb(tb)
    first = e.args[0] if len(e.args) > 0 else None
    iss = isinstance(first, str)
    args_table = H.table()
    for i in range(1 if iss else 0, len(e.args)):
        arg = e.args[i]
        tr = H.tr(H.td(i), H.td(hrepr(arg)))
        args_table = args_table(tr)

    views = H.tabbedView['hrepr-Exception'](
        H.strong(type(e).__name__),
        f': {first}' if iss else '',
        args_table
    )
    last = entries[-1]
    for entry in entries:
        filename, lineno, funcname, line = entry
        absfile = os.path.abspath(filename)
        view = H.view()
        tab = H.tab(
            f'{funcname}@{os.path.basename(absfile)}'
        )
        snippet = H.codeSnippet(
            src = absfile,
            language = "python",
            line = lineno,
            context = 4
        )
        if filename.startswith('<'):
            snippet = snippet(getattr(e, '__repl_string__', ""))

        pane = H.pane(snippet)
        if entry is last:
            view = view(active = True)
        views = views(view(tab, pane))

    return views


master.hrepr.register(Exception, handle_exception)


class Reader(EventDispatcher):
    def __init__(self, source=sys.stdin):
        super().__init__()
        self.source = source

    def read(self):
        line = self.source.readline()
        return self.parse(line)

    def parse(self, line):
        cmd = Props(json.loads(line))
        self.emit(cmd.command, cmd)
        return cmd

    def run(self):
        for line in self.source:
            self.parse(line)
