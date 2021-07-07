import ast
import inspect
from ast import NodeTransformer
from contextlib import contextmanager
from itertools import count
from textwrap import dedent

from _pytest.assertion.rewrite import AssertionRewriter

from myia.abstract import data


class AssertTransformer(NodeTransformer):
    def visit_FunctionDef(self, node):
        newfns = []
        for i, stmt in enumerate(node.body):
            if not isinstance(stmt, ast.Assert):
                raise Exception(
                    "@one_test_per_assert requires all statements to be asserts"
                )
            else:
                newfns.append(
                    ast.FunctionDef(
                        name=f"{node.name}_assert{i + 1}",
                        args=node.args,
                        body=[stmt],
                        decorator_list=node.decorator_list,
                        returns=node.returns,
                    )
                )
        return ast.Module(body=newfns, type_ignores=[])


def one_test_per_assert(fn):
    src = dedent(inspect.getsource(fn))
    filename = inspect.getsourcefile(fn)
    tree = ast.parse(src, filename)
    tree = tree.body[0]
    assert isinstance(tree, ast.FunctionDef)
    tree.decorator_list = []
    new_tree = AssertTransformer().visit(tree)
    ast.fix_missing_locations(new_tree)
    _, lineno = inspect.getsourcelines(fn)
    ast.increment_lineno(new_tree, lineno - 1)
    # Use pytest's assertion rewriter for nicer error messages
    AssertionRewriter(filename, None, None).run(new_tree)
    new_fn = compile(new_tree, filename, "exec")
    glb = fn.__globals__
    exec(new_fn, glb, glb)
    return None


@contextmanager
def predictable_placeholders():
    old_id = data._id
    data._id = count(1)
    try:
        yield
    finally:
        data._id = old_id
