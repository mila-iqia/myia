"""Definitions for Myia's abstract data structures."""

import os
from itertools import count

from hrepr import pstr

from ..utils.intern import Atom, AttrEK, Interned, PossiblyRecursive
from ..utils.misc import Named, myia_hrepr_resources

_id = count(1)
assets = os.path.join(os.path.dirname(__file__), "..", "assets")


#############
# TrackDict #
#############


class Cachable:
    """Base class for objects in which cached transforms can be stored.

    This class is checked for in abstract_map and similar functions to
    determine whether the result of the map/transform can be stored as
    an extra field in the object.
    """


ABSENT = Named("ABSENT")
ANYTHING = Named("ANYTHING")


class Tracks:
    """Collection of properties for an abstract object.

    The main "tracks" (each track is a kind of property) are:

    * "value" -> ValueTrack for the value, if present
    * "interface" -> InterfaceTrack for the Python type that encapsulates
        the interface of the object being represented

    Note that there are two ways to get a track, one that returns the value
    directly, the other that returns a Track object:

    * tracks_obj.interface => type
    * tracks_obj.get_track("interface") => InterfaceTrack(type)
      * tracks_obj.values() or .items() also return Track objects.
    """

    track_handlers = {}

    @classmethod
    def register_track(cls, track_name, track_handler):
        """Register a new track under the given name.

        This adds the corresponding property to Tracks instances. The
        property returns the value of a track rather than a Track instance,
        but the Track is returned by `get_track`, `values` and `items`.
        """

        @property
        def prop(self):
            return self.get_track(track_name).value

        setattr(cls, track_name, prop)
        cls.track_handlers[track_name] = track_handler
        return track_handler

    def __init__(self, tracks=None, **kwargs):
        """Initialize Tracks.

        Arguments:
            tracks: The dictionary of tracks to use, mapping names to
                Track objects.
            kwargs: A dictionary mapping names to values, which will be
                wrapped into Track objects.
        """
        if tracks is not None:
            self._tracks = tracks
        else:
            self._tracks = {}
        self._tracks.update(
            {k: self.track_handlers[k](v) for k, v in kwargs.items()}
        )

    def get_track(self, name):
        """Return the Track for the given track name."""
        return (
            self._tracks[name]
            if name in self._tracks
            else self.track_handlers[name].default()
        )

    def items(self):
        """Return the track_name: Track mapping."""
        return self._tracks.items()

    def values(self):
        """Return a collection of Tracks."""
        return self._tracks.values()

    def __hrepr_short__(self, H, hrepr):
        if hrepr.config.bare_tracks:
            return H.atom()
        else:
            return H.atom("<Tracks>")

    def __hrepr__(self, H, hrepr):
        omit = hrepr.config.omit_tracks or set()
        return H.instance(
            *[
                H.pair(k, hrepr(v.value), delimiter=" ↦ ")
                for k, v in sorted(self._tracks.items())
                if k not in omit
            ],
            vertical=True,
            type="TrackDict",
        )

    __str__ = __repr__ = pstr


class Track:
    """Encapsulates a particular property on an abstract object.

    Subclasses of Track implement different kinds of properties.
    """

    @classmethod
    def default(cls):
        """Return a default Track instance."""
        return cls(ABSENT)

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return type(self) is type(other) and other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __str__(self):
        return f"{type(self).__name__}({self.value})"

    __repr__ = __str__


class ValueTrack(Track):
    """Represents the actual value of an abstract object."""

    @classmethod
    def default(cls):
        """Return the default ValueTrack with value ANYTHING."""
        return cls(ANYTHING)


class InterfaceTrack(Track):
    """Represents the Python type of an abstract object."""


Tracks.register_track("value", ValueTrack)
Tracks.register_track("interface", InterfaceTrack)
Tracks.register_track("ndim", ValueTrack)
Tracks.register_track("shape", ValueTrack)


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

    def __eqkey__(self):
        return Atom(self, tuple(sorted(self.tracks.items())))

    __hrepr_resources__ = myia_hrepr_resources

    def __hrepr_short__(self, H, hrepr):
        name = getattr(
            self.tracks.interface, "__name__", str(self.tracks.interface)
        )
        return H.instance["myia-abstract"](
            type=f"*{name}",
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
            interface = self.tracks.interface
            return H.instance["myia-abstract-structure"](
                *map(hrepr, self.elements),
                *hrepr(
                    self.tracks, omit_tracks={"interface"}, bare_tracks=True
                ).children,
                type=f"*{interface.__name__ if isinstance(interface, type) else type(interface).__name__}",
            )


class AbstractUnion(AbstractValue):
    """Base class for abstract values that are sum types."""

    def __init__(self, options, tracks):
        super().__init__(tracks)
        self.options = list(options)

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


class AbstractDict(AbstractStructure):
    """Represents a dict type.

    Store succession of key-value types in a flat list.
    """

    def __init__(self, items):
        assert not len(items) % 2
        super().__init__(items, {"interface": dict})

    @property
    def keys(self):
        """Get list of key types."""
        return self.elements[::2]

    @property
    def values(self):
        """Get list of value types."""
        return self.elements[1::2]


##################
# Function types #
##################


class AbstractFunction(AbstractStructure):
    """Represents a simple function type."""

    @property
    def args(self):
        """Return the argument types."""
        return self.elements[:-1]

    @property
    def out(self):
        """Return the output type."""
        return self.elements[-1]


####################
# Incomplete types #
####################


class GenericBase(Cachable):
    """Represents a generic type.

    A GenericBase is meant to be unified with an AbstractValue eventually.
    """

    __hrepr_resources__ = myia_hrepr_resources
    __str__ = __repr__ = pstr


class Generic(GenericBase):
    """Represents any type, for use in type signatures.

    Attributes:
        name: The name of the generic. Serves no other purpose than display.
            Different generics can have the same name.
    """

    def __init__(self, name):
        self.name = name

    def __hrepr_short__(self, H, hrepr):
        return H.atom["myia-generic"](f"?{self.name}")


class CanonGeneric(GenericBase):
    """Subtype of GenericBase used to create canonical signatures.

    Equal if they have the same key: CanonGeneric(x) == CanonGeneric(x)
    """

    def __init__(self, rank):
        self.rank = rank

    def __eq__(self, other):
        return type(other) is type(self) and other.rank == self.rank

    def __hash__(self):
        return hash(self.rank)

    def __hrepr_short__(self, H, hrepr):
        return H.atom["myia-canon-generic"](f"?{self.rank}")


class Placeholder(GenericBase):
    """Subtype of GenericBase used as placeholders when processing a node.

    Placeholders are never equal to each other.
    """

    def __init__(self):
        self.id = next(_id)

    def __hrepr_short__(self, H, hrepr):
        return H.atom["myia-placeholder"](f"??{self.id}")
