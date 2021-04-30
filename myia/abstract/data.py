from itertools import count

from hrepr import pstr

from ..utils.intern import Atom, AttrEK, Interned, PossiblyRecursive
from ..utils.misc import Named

_id = count(1)


#############
# TrackDict #
#############


class Cachable:
    pass


ABSENT = Named("ABSENT")
ANYTHING = Named("ANYTHING")


class Tracks:
    track_handlers = {}

    @classmethod
    def register_track(cls, track_name, track_handler):
        @property
        def prop(self):
            return self.get_track(track_name).value

        setattr(cls, track_name, prop)
        cls.track_handlers[track_name] = track_handler
        return track_handler

    def __init__(self, tracks=None, **kwargs):
        if tracks is not None:
            self._tracks = tracks
        else:
            self._tracks = {}
        self._tracks.update(
            {k: self.track_handlers[k](v) for k, v in kwargs.items()}
        )

    def get_track(self, name):
        if name not in self._tracks:
            self._tracks[name] = self.track_handlers[name].default()
        return self._tracks[name]

    def items(self):
        return self._tracks.items()

    def values(self):
        return self._tracks.values()

    def __hrepr_short__(self, H, hrepr):
        if hrepr.config.bare_tracks:
            return H.span()
        else:
            return H.span("<Tracks>")

    def __hrepr__(self, H, hrepr):
        omit = hrepr.config.omit_tracks or set()
        return H.instance(
            *[
                H.pair(k, hrepr(v.value), delimiter=" ↦ ")
                for k, v in self._tracks.items()
                if k not in omit
            ],
            vertical=True,
            type="TrackDict",
        )

    __str__ = __repr__ = pstr


class Track:
    @classmethod
    def default(cls):
        return cls(ABSENT)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(self) is type(other) and other.value == self.value

    def __hash__(self):
        return hash(self.value)


class ValueTrack(Track):
    @classmethod
    def default(cls):
        return cls(ANYTHING)


class InterfaceTrack(Track):
    pass


Tracks.register_track("value", ValueTrack)
Tracks.register_track("interface", InterfaceTrack)


#################
# Abstract data #
#################


class AbstractValue(Interned, PossiblyRecursive, Cachable):
    """Base class for all abstract values.

    Attributes:
        values: A dictionary mapping a Track like VALUE or TYPE
            to a value for that track. Different abstract structures
            may have different tracks, e.g. SHAPE for arrays.
    """

    __cache_eqkey__ = True

    def __init__(self, tracks):
        """Initialize an AbstractValue."""
        super().__init__()
        if not isinstance(tracks, Tracks):
            tracks = Tracks(**tracks)
        self.tracks = tracks

    @property
    def t(self):
        return self.tracks

    def __eqkey__(self):
        return Atom(self, tuple(sorted(self.tracks.items())))

    @classmethod
    def __hrepr_resources__(self, H):
        return H.style(
            """
            .hrepr-instance.myia-abstract {
                margin: 2px;
                background: purple;
                border-color: purple;
            }
            .hrepr-instance.myia-abstract-structure {
                margin: 2px;
                background: blue;
                border-color: blue;
            }
            .hrepr-instance.myia-abstract-union {
                margin: 2px;
                background: darkred;
                border-color: darkred;
            }
            """
        )

    def __hrepr_short__(self, H, hrepr):
        return H.instance["myia-abstract"](
            type=f"*{self.tracks.interface.__name__}",
            short=True,
        )

    def __hrepr__(self, H, hrepr):
        if list(self.tracks._tracks.keys()) == ["interface"]:
            return NotImplemented
        else:
            ifc = self.tracks.interface
            typ = type(self).__name__ if ifc is ABSENT else f"*{ifc.__name__}"
            return H.instance["myia-abstract"](
                *hrepr(
                    self.tracks, omit_tracks={"interface"}, bare_tracks=True
                ).children,
                type=typ,
            )

    __str__ = __repr__ = pstr


class AbstractAtom(AbstractValue):
    """Base class for abstract values that are not structures."""


class AbstractStructure(AbstractValue):
    """Base class for abstract values that are product types."""

    def __init__(self, elements, tracks):
        super().__init__(tracks)
        self.elements = list(elements)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "elements"))

    def __hrepr__(self, H, hrepr):
        if not hasattr(self, "elements"):
            return H.instance["myia-abstract-structure"](
                type=f"{type(self).__name__}"
            )
        else:
            return H.instance["myia-abstract-structure"](
                *map(hrepr, self.elements),
                *hrepr(
                    self.tracks, omit_tracks={"interface"}, bare_tracks=True
                ).children,
                type=f"*{self.tracks.interface.__name__}",
            )


class AbstractUnion(AbstractValue):
    """Base class for abstract values that are sum types."""

    def __init__(self, options, tracks):
        super().__init__(tracks)
        self.options = list(options)  # Possibilities(options)

    def __eqkey__(self):
        v = AbstractValue.__eqkey__(self)
        return AttrEK(self, (v, "options"))

    def __hrepr_short__(self, H, hrepr):
        return H.instance["myia-abstract-union"](
            type="*U",
            short=True,
        )

    def __hrepr__(self, H, hrepr):
        if not hasattr(self, "options"):
            return H.instance["myia-abstract-union"](
                type=f"{type(self).__name__}"
            )
        else:
            return H.instance["myia-abstract-union"](
                *map(hrepr, self.options),
                *hrepr(
                    self.tracks, omit_tracks={"interface"}, bare_tracks=True
                ).children,
                type="*U",
            )


####################
# Incomplete types #
####################


class Generic(Cachable):
    pass


class Opaque(Generic):
    def __init__(self, rank):
        self.rank = rank

    def __eq__(self, other):
        return type(other) is type(self) and other.rank == self.rank

    def __hash__(self):
        return hash(self.rank)

    @classmethod
    def __hrepr_resources__(self, H):
        return H.style(
            """
            .myia-opaque {
                color: white;
                background: black;
                padding: 2px;
                margin: 2px;
                font-weight: bold;
            }
            """
        )

    def __hrepr_short__(self, H, hrepr):
        return H.atom["myia-opaque"](f"?{self.rank}")

    __str__ = __repr__ = pstr


class Placeholder(Generic):
    def __init__(self):
        self.id = next(_id)

    @classmethod
    def __hrepr_resources__(self, H):
        return H.style(
            """
            .myia-placeholder {
                color: white;
                background: brown;
                padding: 2px;
                margin: 2px;
                font-weight: bold;
            }
            """
        )

    def __hrepr_short__(self, H, hrepr):
        return H.atom["myia-placeholder"](f"??{self.id}")

    __str__ = __repr__ = pstr
