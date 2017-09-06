"""
Helpers to track the evolution of a node through transformations,
notably source code location.

Consider:

    with About(original_node, 'my-transform'):
        return f(original_node)

Any node (as defined in myia.stx.nodes) created while the ``with``
block is active, in this case during the execution of ``f``, will have
its ``about`` field set to the ``About`` instance. This will mark it
as being "about" ``original_node``, with the tag ``my-transform``. The
``about`` field can thus form a chain going all the way back to the
original node made during parsing.

All code transforms should use ``with About(...)`` in their implementation
in order to preserve this chain, with as fine a granularity as possible.
"""


from typing import Any
import threading
import textwrap


class MyiaSyntaxError(Exception):
    """
    Class for syntax errors in Myia. This exception type should be
    raised for any feature that is not supported.

    Attributes:
        message: A precise assessment of the problem.
        location: The error's location in the original source. If not
            provided, it will be extracted from the node we are currently
            About.
    """
    def __init__(self, message: str, location: 'Location' = None) -> None:
        self.location = location or current_location()
        self.message = message
        super().__init__(self.message, self.location)


# We use per-thread storage for the about stack.
_about = threading.local()
_about.stack = [None]


def top():
    """
    Return the currently active ``About`` instance.
    """
    return _about.stack[-1]


def current_location():
    """
    Follow the about chain of the node we are currently about in order
    to find where in the source code it ultimately came from.
    """
    abt = top()
    while abt:
        node = abt.node
        if getattr(node, 'about', None):
            abt = node.about
        else:
            abt = None
    if isinstance(node, Location):
        return node
    return None


class About:
    """
    Any code executed during the ``About`` context manager is understood
    to be "about" the node given in the constructor. It is up to
    compliant object constructors to consult what node they are about and
    to store that information. This is done by ``MyiaASTNode`` instances.

    ``About`` instances are pushed on a per-thread stack, therefore
    ``with`` calls to ``About`` can be nested.
    """
    def __init__(self, node, transform):
        self.node = node
        self.transform = transform

    def __enter__(self):
        _about.stack.append(self)

    def __exit__(self, etype, evalue, etraceback):
        _about.stack.pop()


class AboutPrinter:
    """
    Through its ``__hrepr__`` representation, an AboutPrinter can be
    printed out to Buche and will display a chain of about nodes that
    can be navigated, along with stack traces for the creation of each
    node when they are available.
    """
    def __init__(self, node):
        self.node = node

    def node_hrepr(self, node, H, hrepr):
        if hasattr(node, 'trace'):
            views = H.tabbedView()
            views = views(H.view(H.tab('node'), H.pane(hrepr(node))))
            views = views(H.view(H.tab('trace'), H.pane(hrepr(node.trace))))
            return views
        else:
            return hrepr(node)

    def __hrepr__(self, H, hrepr):
        views = H.tabbedView()
        node = self.node

        nodes = [self.node_hrepr(node, H, hrepr)]
        transforms = []

        while node and getattr(node, 'about', None):
            about = node.about
            node = about.node
            nodes.append(self.node_hrepr(node, H, hrepr))
            transforms.append(about.transform)
        transforms.append('orig')

        for transform, node in reversed(list(zip(transforms, nodes))):
            tab = H.tab(transform)
            pane = H.pane(node)
            views = views(H.view(tab, pane))

        return views


class Location:
    """
    Represents a source code location for an AST node.

    Attributes:
        url (str): The path of the code file.
        line (int): The line number in that file.
        column (int): The column number in that file.
    """

    def __init__(self,
                 url: str,
                 line: int,
                 column: int,
                 node: Any = None) -> None:
        self.url = url
        self.line = line
        self.column = column
        self.node = node

    def traceback(self) -> str:
        """
        Print out a "traceback" that corresponds to this location,
        with the line printed out and a caret at the right column.
        Basically:

        >>> loc.traceback()
          File {url}, line {line}, column {column}
            x = f(y)
                ^

        This is mostly meant for printing out ``MyiaSyntaxError``s.
        """
        try:
            with open(self.url) as file:
                raw_code = file.readlines()[self.line - 1].rstrip("\n")
                raw_caret = ' ' * self.column + '^'
                code, caret = textwrap.dedent(
                    raw_code + '\n' + raw_caret
                ).split("\n")
            return '  File "{}", line {}, column {}\n    {}\n    {}'.format(
                self.url, self.line, self.column, code, caret)
        except FileNotFoundError:
            return '  File "{}", line {}, column {}'.format(
                self.url, self.line, self.column)

    def __str__(self) -> str:
        return '{}@{}:{}'.format(self.url, self.line, self.column)

    def __hrepr__(self, H, hrepr):
        return H.codeSnippet(
            src = self.url,
            language = "python",
            line = self.line,
            column = self.column + 1,
            context = 4
        )
