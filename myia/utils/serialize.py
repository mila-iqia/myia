"""Serialization utilities for graphs and their properties."""

try:
    from yaml import CSafeLoader as SafeLoader, CSafeDumper as SafeDumper
except ImportError:  # pragma: no cover
    from yaml import SafeLoader, SafeDumper

import sys
from dataclasses import is_dataclass


class MyiaDumper(SafeDumper):
    """Customize the dumper."""

    def __init__(self, stream):
        """Record stream, even for C."""
        super().__init__(stream, encoding='utf-8', explicit_end=True)
        self.stream = stream

    def represent(self, data):
        """Represent and flush."""
        super().represent(data)
        self.stream.flush()


class MyiaLoader(SafeLoader):
    """Customize the loader."""
    def __init__(self, stream):
        stream.read = stream.read1
        stream.readinto = stream.readinto1
        super().__init__(stream)

    def construct_document(self, node):
        """Add support for finalizers."""
        self._finalizers = []
        res = super().construct_document(node)
        for f in self._finalizers:
            f()
        self._finalizers = []
        return res

    def add_finalizer(self, f):
        """Regsiter a finalizer to be run when the loading is finished."""
        assert callable(f)
        self._finalizers.append(f)


def __serialize__(self, dumper):
    data = self._serialize()
    assert isinstance(data, dict)
    return dumper.represent_mapping(getattr(self, '@SERIAL_TAG'), data)


def __serialize_seq__(self, dumper):
    seq = self._serialize()
    assert isinstance(seq, (tuple, list))
    return dumper.represent_sequence(getattr(self, '@SERIAL_TAG'), seq)


def __serialize_scal__(self, dumper):
    scal = self._serialize()
    assert scal is not None
    assert isinstance(scal, str)
    return dumper.represent_scalar(getattr(self, '@SERIAL_TAG'), scal)


def __serialize_dc__(self, dumper):
    data = dict((k, getattr(self, k))
                for k in self.__dataclass_fields__.keys())
    return dumper.represent_mapping(getattr(self, '@SERIAL_TAG'), data)


def _serialize(dumper, self):
    return self.__serialize__(dumper)


def _make_construct(cls, dc, sequence, scalar):
    def _construct(loader, node):
        it = cls._construct()
        yield next(it)
        data = loader.construct_mapping(node)
        try:
            it.send(data)
        except StopIteration as e:
            if e.value is not None:
                loader.add_finalizer(e.value)
    if dc:
        if cls.__dataclass_params__.frozen:
            def _construct(loader, node):  # noqa: F811
                data = loader.construct_mapping(node)
                return cls(**data)
        else:
            breakpoint()
    if sequence:
        def _construct(loader, node):  # noqa: F811
            it = cls._construct()
            yield next(it)
            data = loader.construct_sequence(node)
            try:
                it.send(data)
            except StopIteration as e:
                return e.value
    if scalar:
        def _construct(loader, node):  # noqa: F811
            data = loader.construct_scalar(node)
            return cls._construct(data)
    return _construct


def serializable(tag, *, dc=None, sequence=False, scalar=False):
    """Class decorator to make the wrapped class serializable.

    Parameters:
        tag: string, serialization tag, must be unique
        dc: bool, class is a dataclass (autodetected, but can override)
        sequence: _serialize returns a sequence (tuple or list)
        scalar: _serialize returns a single item.

    """
    def wrapper(cls):
        nonlocal dc
        if dc is None and is_dataclass(cls):
            dc = True
        setattr(cls, '@SERIAL_TAG', tag)
        if not hasattr(cls, '__serialize__'):
            method = __serialize__
            if dc:
                method = __serialize_dc__
            if sequence:
                method = __serialize_seq__
            if scalar:
                method = __serialize_scal__
            setattr(cls, '__serialize__', method)
        _construct = _make_construct(cls, dc, sequence, scalar)
        MyiaDumper.add_representer(cls, _serialize)
        MyiaLoader.add_constructor(tag, _construct)
        return cls
    return wrapper


def _serialize_tuple(dumper, data):
    return dumper.represent_sequence('tuple', data)


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


MyiaDumper.add_representer(tuple, _serialize_tuple)
MyiaLoader.add_constructor('tuple', _construct_tuple)

_OBJ_MAP = {}
_TAG_MAP = {}


def _serialize_unique(dumper, obj):
    tag = _OBJ_MAP.get(obj, None)
    if tag is None:
        return dumper.represent_undefined(obj)
    else:
        return dumper.represent_scalar(tag, '')


def _construct_unique(loader, node):
    obj = _TAG_MAP.get(node.tag, None)
    if obj is None:
        return loader.construct_undefined(node)
    else:
        return obj


MyiaDumper.add_representer(None, _serialize_unique)
MyiaLoader.add_constructor(None, _construct_unique)


def register_serialize(obj, tag):
    """Serialize unique objects.

    The object instance will be associated with the given tag.
    """
    assert isinstance(tag, str)
    assert obj not in _OBJ_MAP
    assert tag not in _TAG_MAP
    _OBJ_MAP[obj] = tag
    _TAG_MAP[tag] = obj


def dump(o, stream=sys.stdout):
    """Dump the passed-in object to the specified stream."""
    dumper = MyiaDumper(stream)
    try:
        dumper.open()
        dumper.represent(o)
        dumper.close()
    finally:
        dumper.dispose()


def load(stream):
    """Load one object from the stream."""
    loader = MyiaLoader(stream)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()
