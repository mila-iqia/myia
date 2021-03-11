import ast
import inspect
from ast import NodeTransformer
from textwrap import dedent

from _pytest.assertion.rewrite import AssertionRewriter
from ovld import ovld

from myia.abstract import data
from myia.abstract.to_abstract import precise_abstract


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


@precise_abstract.variant
def _to_abstract(self, x: type):
    return data.AbstractAtom({"interface": x})


@ovld
def _to_abstract(self, x: (data.Generic, data.AbstractValue)):
    return x


def A(*args):
    if len(args) == 1:
        arg = args[0]
    else:
        arg = args
    return _to_abstract(arg)


def Un(*opts):
    return data.AbstractUnion([A(opt) for opt in opts], tracks={})
