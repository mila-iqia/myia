
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

    def show(self, obj, path='/', **hrepr_params):
        r = self.hrepr(obj)  # , **hrepr_params)
        for res in self.hrepr.resources:
            if res not in self.resources:
                self.log(res,
                         path='/',
                         format='html')
                self.resources.add(res)
        self.log(r, format='html', path=path)


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

    def open(self, name, type, **params):
        self.to_open[name] = (type, params)

    def join_path(self, p):
        return f'{self.channel.rstrip("/")}/{p.rstrip("/")}'

    def __getitem__(self, item):
        subchannel = self.join_path(item)
        info = self.to_open.get(item, None)
        if info:
            del self.to_open[item]
            type, params = info
            self.master.raw(command='open',
                            path=subchannel,
                            type=type,
                            **params)
        return Buche(self.master, subchannel)

    def __call__(self, obj, **params):
        self.master.show(obj, hrepr_params=params, path=self.channel)


master = MasterBuche(StdHRepr())
buche = Buche(master, '/')
