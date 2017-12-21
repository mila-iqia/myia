"""Base interface for graph-based IR.

This module defines the basic interface for the graph-based IR(s) used in Myia.

"""
from abc import ABC, abstractmethod
from typing import Iterable


class Node(ABC):
    """A node in a graph IR.

    Attributes:
        incoming: A collection of nodes which are inputs to this node.
        outgoing: A collection of nodes that use this node as an input.

    """

    @property
    @abstractmethod
    def incoming(self) -> Iterable['Node']:
        """Return nodes used as inputs."""
        pass

    @property
    @abstractmethod
    def outgoing(self) -> Iterable['Node']:
        """Return nodes which use this node as an input."""
        pass
