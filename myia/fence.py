"""Check Python code for compatibility with Myia."""
import ast
from typing import Sequence


class MyiaSyntaxError(SyntaxError):
    """Syntax error for Python code unsupported by Myia.

    This class inherits from `SyntaxError` and pretty prints the relevant line
    and code when raised.

    """

    def __init__(self, filename: str, lineno: int, offset: int, text: str,
                 message: str = None) -> None:
        """Construct a Myia syntax error.

        Args:
            filename: The filename from which the code was loaded. Use a string
                like `<string>` for code that was not loaded from a file.
            lineno: The 1-indexed line in the file or string with the offending
                expressiong or statement.
            offset: The column which should be pointed to in the error message.
            text: The text to print, which should normally be the line in the
                file given bye `lineno` and `filename`.
            message: An optional message to clarify what is invalid about this
                syntax.

        """
        full_message = "invalid syntax"
        if message:
            full_message = f"{full_message}: {message}"
        super().__init__(full_message)
        self.filename = filename
        self.lineno = lineno
        self.offset = offset
        self.text = text


class Fence(ast.NodeVisitor):
    """Check if each Python AST node is supported.

    This node visitor walks the Python AST, and raises a syntax error for each
    AST node that is not explicitly supported.

    If a node is fully supported, it can be added to the `supported` attribute.
    If a node is partially supported, implement a `visit_Node` method which
    calls the `raise` method when certain conditions are not met.

    Attributes:
        supported: A container of nodes that are fully supported.

    """

    supported = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg)

    def __init__(self, filename: str, lines: Sequence[str],
                 line_offset: int) -> None:
        """Construct a fence.

        Args:
            filename: The name of the file from which the AST was loaded.
            lines: A sequence of lines whose indices correspond to the line
                numbers in the AST.
            line_offset: The 1-indexex line number in the file on which the
                parsed AST starts. This allows line numbers in the error
                message to be correctly reported.

        """
        self.filename = filename
        self.lines = lines
        self.line_offset = line_offset - 1

        self.lineno = 1
        self.col_offset = 0

        # Deal with indented code that was dedented for parsing
        first_line = lines[0].expandtabs()
        self.dedent = len(first_line) - len(first_line.lstrip())

    def visit(self, node: ast.AST):
        """Visit a node."""
        if isinstance(node, (ast.expr, ast.stmt)):
            self.lineno = node.lineno
            self.col_offset = node.col_offset
        if isinstance(node, self.supported):
            self.generic_visit(node)
        else:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, None)
            if visitor:
                return visitor(node)
            self.raise_()

    def visit_Pass(self, node: ast.Pass):
        """No pass."""
        self.raise_("you shall not pass")

    def raise_(self, message: str = None):
        """Raise a syntax error for an unsupported node.

        Args:
            message: An additional message to add to the exception.

        """
        raise MyiaSyntaxError(self.filename, self.lineno + self.line_offset,
                              self.col_offset + self.dedent,
                              self.lines[self.lineno - 1], message)


if __name__ == '__main__':
    def f(x):
        pass

    import ast
    import inspect
    import textwrap
    fence = Fence(inspect.getsourcefile(f), *inspect.getsourcelines(f))
    fence.visit(ast.parse(textwrap.dedent(inspect.getsource(f))))
