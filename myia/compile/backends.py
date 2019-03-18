class UnknownBackend(Exception):
    """Indicates that the backend name is not recognized."""

    
class LoadingError(Exception):
    """Indicates that there was an error loading the backend.
    
    This can happen because of missing dependencies.  There should be
    a chained exception with the original error that comes with it.
    """


_backends = {
    'debug': None,
    'nnvm': None,
}


def load_backend(name):
    """Load the named backend.

    Raises:
        UnknownBackend: The name is not recognized.
        LoadingError: There was an error loading the backend.
    """    


class Backend:
    """This is a base class that all backends should implement."""

    def compile(self, graph):
        """Compile the group of graphs rooted at `graph`.

        This function takes in a fully typed graph cluster rooted at
        `graph` with a manager and must return a callable that accepts
        arguments of the same type and number as the root graph.
        """
        pass

    def to_tensor(self, dlp):
        """
        Convert a value from a DLpack PyCapsule to a backend-appropriate value.
        """
        pass

    def to_dlpack(self, v):
        """
        Convert a backend-specific tensor to a DLpack PyCapsule.
        """
        pass
