"""Compilation backends."""

import os
import urllib
import weakref

from ... import abstract, xtype
from ..channel import RPCProcess, handle
from ..transform import convert_grad


class UnknownBackend(Exception):
    """Indicates that the backend name is not recognized."""


class LoadingError(Exception):
    """Indicates that there was an error loading the backend.

    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


def channel_loader(pkg, name):
    """Helper function for simple backends.

    This will return a callable that will load a module, retrieve a
    object from its namespace and return that.

    """
    def loader(init_args):
        proc = RPCProcess(pkg, name, init_args)
        return ChannelBackend(proc)
    return loader


def relay_defaults(target='cpu', device_id=0):
    """Format options for relay."""
    return dict(target=target, device_id=device_id)


def pytorch_default(device='cpu:0'):
    """Format options for pytorch."""
    if device == 'cuda':
        device = 'cuda:0'
    if device == 'cpu':
        device = 'cpu:0'
    return dict(device=device)


_backends = {
    'relay': (channel_loader('myia.compile.backends.relay', 'RelayBackendR'),
              relay_defaults),
    'pytorch': (channel_loader('myia.compile.backends.pytorch',
                               'PyTorchBackendR'), pytorch_default)
}

_active_backends = weakref.WeakValueDictionary()


def get_default():
    """Returns the default backend.

    This is fetched from the MYIA_BACKEND environement variable or
    from the built-in defaults.

    The syntax for specifiying a backend is
    'name?option1=value1&option2=value2' when name is the name of the
    backend and option1 is a valid keyword option for that backend.
    This is strongly inspired by HTTP query syntax except you don't
    need to urlencode values.

    As an example here is a string for pytorch on the GPU

        pytorch?target=cuda:0

    """
    backend, opts = parse_default()
    assert backend is not None
    return load_backend(backend, opts)


def parse_default():
    """Parses the default backend.

    Returns name and options from the environement or builtin default.
    See the documentation of get_default() for the backend string syntax.
    """
    backend_spec = os.environ.get('MYIA_BACKEND', 'pytorch')
    backend, *opts = backend_spec.split('?', maxsplit=1)
    if len(opts) == 1:
        opts = urllib.parse.parse_qs(opts[0], keep_blank_values=True,
                                     strict_parsing=True, errors='strict')
        for k in opts:
            assert len(opts[k]) == 1
            opts[k] = opts[k][0]
    else:
        assert len(opts) == 0
        opts = {}
    return backend, opts


def load_backend(name, options=None):
    """Load the named backend.

    Returns the backend class registered for the name.

    If you pass None as the name, this will load the default backend.
    See the documenation for get_default() for more information.

    Raises:
        UnknownBackend: The name is not recognized.
        LoadingError: There was an error loading the backend.

    """
    if name is None:
        assert options is None
        return get_default()
    if options is None:
        options = {}
    if name not in _backends:
        raise UnknownBackend(name)
    options = _backends[name][1](**options)
    key = (name, tuple(sorted(list(options.items()))))
    res = _active_backends.get(key, None)
    if res is None:
        try:
            res = _backends[name][0](options)
            _active_backends[key] = res
        except Exception as e:
            raise LoadingError(name) from e
    return res


def register_backend(name, load_fn, defaults_fn):
    """Register a new backend.

    This is to allow third party libraries to register their own
    backends if loaded by the user.  Built-in backends are
    preregistered.

    Arguments:
        name (str): Name of the backend, must be unique
        load_fn: function that will load the backend.  This must
                 return a callable that will take keyword arguemnts
                 for options.
        defaults_fn: function that takes the same default arguments as
                     load_fn and maps them to canonical and/or default
                     values.

    """
    assert name not in _backends
    _backends[name] = (load_fn, defaults_fn)


class Backend:
    """This is a class interface that all backends must implement."""

    def compile(self, graph, argspec, outspec):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        raise NotImplementedError('compile')

    def from_backend_value(self, v, t):
        """Convert a backend value to an intermediate value."""
        raise NotImplementedError('from_backend_value')

    def to_backend_value(self, v, t):
        """Convert an intermediate value to a backend value."""
        raise NotImplementedError('to_backend_value')


class Converter:
    """Converts values between representations for backends."""

    def convert_array(self, v, t):
        """Converts array values."""
        raise NotImplementedError("convert_numpy")

    def convert_scalar(self, v, t):
        """Convert numeric scalars."""
        raise NotImplementedError("convert_scalar")

    def convert_nil(self, v, t):
        """Convert Nil values."""
        raise NotImplementedError("convert_nil")

    def convert_dead(self, v, t):
        """Convert dead values."""
        raise NotImplementedError("convert_dead")

    def convert_bool(self, v, t):
        """Convert boolean values."""
        raise NotImplementedError("convert_bool")

    def convert_env(self, v, t):
        """Convert a grad env."""
        raise NotImplementedError("convert_env")

    def convert_universe(self, v, t):
        """Convert a Universe."""
        raise NotImplementedError("convert_universe")

    def convert_handle(self, v, t):
        """Convert a Handle."""
        raise NotImplementedError("convert_handle")

    def convert_tuple(self, v, t):
        """Convert a tuple."""
        raise NotImplementedError("convert_tuple")

    def convert_tagged(self, v, t):
        """Convert a union value."""
        raise NotImplementedError("convert_tagged")

    def convert_type(self, v, t):
        """Convert a type value."""
        raise NotImplementedError("convert_type")

    def convert_random_state(self, v, t):
        """Convert a random state value."""
        raise NotImplementedError("convert_type")

    def __call__(self, v, t):
        """Convert a value."""
        if v is abstract.DEAD:
            return self.convert_dead(v, t)
        elif isinstance(t, abstract.AbstractArray):
            return self.convert_array(v, t)
        elif isinstance(t, abstract.AbstractRandomState):
            return self.convert_random_state(v, t)
        elif isinstance(t, abstract.AbstractScalar):
            if issubclass(t.xtype(), xtype.Number):
                return self.convert_scalar(v, t.xtype())
            elif issubclass(t.xtype(), xtype.Bool):
                return self.convert_bool(v, t.xtype())
            elif issubclass(t.xtype(), xtype.Nil):
                return self.convert_nil(v, t.xtype())
            elif issubclass(t.xtype(), xtype.EnvType):
                return self.convert_env(v, t.xtype())
            elif issubclass(t.xtype(), xtype.UniverseType):
                return self.convert_universe(v, t.xtype())
            else:
                raise NotImplementedError(f'convert for scalar {t.xtype()}')
        elif isinstance(t, abstract.AbstractTuple):
            return self.convert_tuple(v, t)
        elif isinstance(t, abstract.AbstractTaggedUnion):
            return self.convert_tagged(v, t)
        elif isinstance(t, abstract.AbstractType):
            return self.convert_type(v, t)
        elif isinstance(t, abstract.AbstractHandle):
            return self.convert_handle(v, t)
        else:
            raise NotImplementedError(f'convert for {t}')


def _close_and_wait(stream):
    stream.close()
    os.waitpid(-1, 0)


class ChannelBackend(Backend):
    """Backend based on a channel to another process."""

    def __init__(self, proc):
        """Remote."""
        self.proc = proc
        weakref.finalize(proc, _close_and_wait, proc.proc.stdin)

    def compile(self, graph, argspec, outspec):
        """Remote."""
        graph = convert_grad(graph)
        return self.proc.call_method('compile', graph, argspec, outspec)

    def from_backend_value(self, v, t):
        """Remote."""
        return self.proc.call_method('from_backend_value', v, t)

    def to_backend_value(self, v, t):
        """Remote."""
        return self.proc.call_method('to_backend_value', v, t)


class HandleBackend(Backend):
    """Proxy for remote process backend."""

    def __init__(self, real):
        """Set the proxied backend."""
        self.real = real

    def compile(self, graph, argspec, outspec):
        """Proxy."""
        return handle(self.real.compile(graph, argspec, outspec))

    def from_backend_value(self, v, t):
        """Remote."""
        return self.real.from_backend_value(v, t)

    def to_backend_value(self, v, t):
        """Remote."""
        return handle(self.real.to_backend_value(v, t))
