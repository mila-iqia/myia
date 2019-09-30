"""Serialization utilities for graphs and their properties."""

# This is a mess of imports

# 1) We prefer ruamel.yaml since it has merged the large files fix.
# 2) Conda has a different name for it since it doesn't like namespaces.
# 3) We attempt to use pyyaml as a fallback
try:  # pragma: no cover
    try:
        from ruamel.yaml import (
            CSafeLoader as SafeLoader,
            CSafeDumper as SafeDumper
        )
    except ImportError:
        try:
            from ruamel_yaml import (
                CSafeLoader as SafeLoader,
                CSafeDumper as SafeDumper
            )
        except ImportError:
            from yaml import (
                CSafeLoader as SafeLoader,
                CSafeDumper as SafeDumper
            )
except ImportError as e:  # pragma: no cover
    raise RuntimeError("""
Couldn't find a C-backed version of yaml.

Please install either ruamel.yaml or PyYAML with the C extension. The python
 versions are just too slow to work properly.
""") from e

import codecs
import os
import traceback
from dataclasses import is_dataclass

import numpy as np


class MyiaDumper(SafeDumper):
    """Customize the dumper."""

    def __init__(self, fd):
        """Record stream, even for C."""
        stream = os.fdopen(fd, 'wb', buffering=0, closefd=False)
        super().__init__(stream, encoding='utf-8', explicit_end=True)
        self.stream = stream

    def represent(self, data):
        """Represent and flush."""
        super().represent(data)
        self.stream.flush()


class MyiaLoader(SafeLoader):
    """Customize the loader."""

    def __init__(self, fd):
        """Make sure reads don't block."""
        stream = os.fdopen(fd, 'rb', buffering=0, closefd=False)
        super().__init__(stream)

    def determine_encoding(self):  # pragma: no cover
        """This is a workaround for the python version of pyyaml.

        It really wants to read from the stream when creating the
        loader object to figure out the encoding.  We just statically
        figure it out here instead.
        """
        self.raw_decode = codecs.utf_8_decode
        self.encoding = 'utf-8'
        self.raw_buffer = b""

    def construct_document(self, node):
        """Add support for finalizers."""
        self._finalizers = []
        res = super().construct_document(node)
        for f in self._finalizers:
            f()
        self._finalizers = []
        return res

    def add_finalizer(self, f):
        """Register a finalizer to be run when the loading is finished."""
        assert callable(f)
        self._finalizers.append(f)


class LoadedError(Exception):
    """Represent an error that was serialized."""

    def __init__(self, message):
        """An error with a single message."""
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return f"LoadedException"


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
        assert cls.__dataclass_params__.frozen

        def _construct(loader, node):  # noqa: F811
            data = loader.construct_mapping(node)
            return cls(**data)
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


def serializable(tag, *, dc=None, sequence=False, scalar=False,
                 construct=True):
    """Class decorator to make the wrapped class serializable.

    Parameters:
        tag: string, serialization tag, must be unique
        dc: bool, class is a dataclass (autodetected, but can override)
        sequence: _serialize returns a sequence (tuple or list)
        scalar: _serialize returns a single item.
        construct: register the deserialization function or not

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
        MyiaDumper.add_representer(cls, _serialize)
        if construct:
            _construct = _make_construct(cls, dc, sequence, scalar)
            MyiaLoader.add_constructor(tag, _construct)
        return cls
    return wrapper


def _serialize_tuple(dumper, data):
    return dumper.represent_sequence('tuple', data)


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


MyiaDumper.add_representer(tuple, _serialize_tuple)
MyiaLoader.add_constructor('tuple', _construct_tuple)


def _serialize_ndarray(dumper, data):
    return dumper.represent_mapping(
        'arraydata', {'dtype': data.dtype.str, 'shape': data.shape,
                      'data': data.tobytes()})


def _construct_ndarray(loader, node):
    data = loader.construct_mapping(node)
    res = np.frombuffer(data['data'], dtype=data['dtype'])
    return res.reshape(data['shape'])


MyiaDumper.add_representer(np.ndarray, _serialize_ndarray)
MyiaLoader.add_constructor('arraydata', _construct_ndarray)


def register_npscalar(tag, cls):
    """Regsiter serialization functions for numpy scalars."""
    def _serialize(dumper, data):
        return dumper.represent_scalar(tag, repr(data))

    def _construct(loader, node):
        return cls(loader.construct_scalar(node))

    MyiaDumper.add_representer(cls, _serialize)
    MyiaLoader.add_constructor(tag, _construct)


register_npscalar('float16', np.float16)
register_npscalar('float32', np.float32)
register_npscalar('float64', np.float64)
register_npscalar('float128', np.float128)

register_npscalar('int8', np.int8)
register_npscalar('int16', np.int16)
register_npscalar('int32', np.int32)
register_npscalar('int64', np.int64)

register_npscalar('uint8', np.uint8)
register_npscalar('uint16', np.uint16)
register_npscalar('uint32', np.uint32)
register_npscalar('uint64', np.uint64)
register_npscalar('bool_', np.bool_)


_OBJ_MAP = {}
_TAG_MAP = {}


def _serialize_unique(dumper, obj):
    tag = _OBJ_MAP.get(obj, None)
    if tag is None:
        if isinstance(obj, Exception):
            return dumper.represent_scalar(
                'error',
                '\n'.join(traceback.format_exception(type(obj), obj,
                                                     obj.__traceback__)))
        elif hasattr(obj, '_serialize_replace'):
            return dumper.represent_data(obj._serialize_replace())
        else:
            return dumper.represent_undefined(obj)
    else:
        return dumper.represent_scalar(tag, '')


def _construct_unique(loader, node):
    if node.tag == 'error':
        data = loader.construct_scalar(node)
        return LoadedError(data)
    obj = _TAG_MAP.get(node.tag, None)
    if obj is None:
        return loader.construct_undefined(node)  # pragma: no cover
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


def dump(o, fd):
    """Dump the passed-in object to the specified stream."""
    dumper = MyiaDumper(fd)
    try:
        dumper.open()
        dumper.represent(o)
        dumper.close()
    finally:
        dumper.dispose()


def load(fd):
    """Load one object from the stream."""
    loader = MyiaLoader(fd)
    try:
        return loader.get_single_data()
    finally:
        loader.dispose()


__consolidate__ = True
__all__ = [
    'LoadedError',
    'MyiaDumper',
    'MyiaLoader',
    'dump',
    'load',
    'register_npscalar',
    'register_serialize',
    'serializable',
]
