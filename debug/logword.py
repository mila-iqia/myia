
from buche import buche
import random
import hashlib
import colorsys
import os


_log = []


def ibuche(*args, **kwargs):
    _log.append(args)
    buche(*args, interactive=True, **kwargs)


class WordGroup:
    """Deterministic word generator.

    Generates a sequence of words from /usr/share/dict/words. The sequence
    deterministically depends on the name of the group.
    """

    _words = []

    @classmethod
    def words(cls):
        if not cls._words:
            cls._words[:] = [
                word.lower()
                for word in open('/usr/share/dict/words').read().split('\n')
            ]
        return cls._words

    def __init__(self, name):
        self.name = name
        self.hash = int(hashlib.md5(self.name.encode()).hexdigest(), base=16)
        self.R = random.Random(self.hash)
        self.current = None

    def gen(self):
        words = self.words()
        w = words[self.R.randint(0, len(words) - 1)]
        self.current = w
        return w

    def rgb(self):
        """Generate an RGB color string for this group."""
        # We generate the color in the YIQ space first because the Y component,
        # corresponding to brightness, is fairly accurate, so we can easily
        # restrict it to a range that looks decent on a white background, and
        # then convert to RGB with the standard colorsys package. The IQ
        # components control hue and have bizarre valid ranges.
        h = self.hash
        # 0.3 <= Y <= 0.6
        y = 0.3 + ((h & 0xFF) / 0xFF) * 0.4
        h >>= 16
        i = (((h & 0xFF) - 0x80) / 0x80) * 0.5957
        h >>= 16
        q = (((h & 0xFF) - 0x80) / 0x80) * 0.5226
        r, g, b = colorsys.yiq_to_rgb(y, i, q)
        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)
        return f'rgb({r}, {g}, {b})'


class Logword:
    """Tool to track progress through a program on stdout."""

    _groups = {}

    def __init__(self,
                 watch=None,
                 gen=True,
                 nowatch_log=False,
                 print_word=True,
                 group='main',
                 data=[]):
        self.data = data if isinstance(data, (list, tuple)) else [data]
        if group not in Logword._groups:
            Logword._groups[group] = WordGroup(group)
        self.group = Logword._groups[group]
        self.watch = watch
        self.word = self.group.gen() if gen else self.group.current
        self.match = self.watch == self.word
        self.active = self.match or (nowatch_log and self.watch is None)
        if print_word:
            self.log(self, *self.data, force=self.watch is None)

    def breakpoint(self):
        if self.active:
            breakpoint()

    def log(self, *objs, force=False):
        if force or self.active:
            if os.environ.get('BUCHE'):
                ibuche(*objs)
            else:
                print(*objs)

    def __bool__(self):
        return self.active

    def __str__(self):
        return f'⏎ {self.group.name}:{self.word}'

    def __hrepr__(self, H, hrepr):
        return H.div(f'⏎ {self.group.name}:{self.word}',
                     style=f'color:{self.group.rgb()};font-weight:bold;')

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass


def logword(*data, **kw):
    """Prints out a word along with the given data."""
    return Logword(data=data, **kw)


def afterword(word, **kw):
    """True after logword prints out the given word."""
    return Logword(watch=word, gen=False, print_word=False, **kw)


def breakword(word, **kw):
    """Activate a breakpoint after logword prints out the given word."""
    afterword(word, **kw).breakpoint()
