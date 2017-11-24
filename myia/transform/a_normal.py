"""
Several code transformations are implemented here.

* ANormalTransformer: Transform AST to A-Normal form.
* CollapseLet: Transform nested Lets into a single Let.
"""


from typing import Union, Any, List, cast, \
    Sequence, Tuple as TupleT, Optional, Callable

import re
from ..stx import \
    MyiaASTNode, ApplyNode as Apply, Symbol, ValueNode as Value, \
    LetNode as Let, LambdaNode as Lambda, ClosureNode as Closure, \
    TupleNode as Tuple, Transformer, GenSym, LHS, Bindings, \
    python_universe, create_lambda, GenSym, nsym, ANORM


# TODO: Lambda should never be a sub-expression, it should be pulled
#       out to the top-level, and wrapped with Closure if needed.


agen = GenSym('global::a_normal')


def a_normal(node: MyiaASTNode) -> MyiaASTNode:
    """
    transform the expression represented by this node in
    A-normal form (ANF). ANF forbids expression nesting,
    so e.g. (f (g x)) would become:
    (let ((tmp (g x))
          (out (f tmp)))
      out)

    More precisely:

    * Only bind each variable once, but that's already done
      by Parser.
    * A leaf node is Symbol or Value.
    * All arguments to Apply, Tuple, Closure must be leaf
      nodes.
    * Begin is reduced to Let.

    The end result is that all functions become a sequence
    of bindings, where each binding is a call that only
    involves bindings, not sub-expressions. They become
    giant Lets.

    ANF is similar to SSA representation.

    Arguments:
        node: The expression to process

    Returns:
        The expression in A-normal form
    """
    assert isinstance(node, Lambda)
    orig = node
    node = ANormalTransformer().transform(node)
    node = CollapseLet().transform(node)
    sym = agen(orig.ref, ANORM)
    python_universe.associate(sym, node)
    return node


StashType = TupleT[Optional[str], Bindings]


class ANormalTransformer(Transformer):
    """
    Transform Myia AST to A-Normal form.

    Don't use this directly, use the ``a_normal`` function.

    This transformer allows nested Lets, but CollapseLet
    can be applied to the result to fix that (which is
    what the a_normal function does.)
    """
    __transform__ = 'a-normal'

    def __init__(self, gen: GenSym = None) -> None:
        self.gen = gen

    def stash(self,
              stash: StashType,
              result: MyiaASTNode,
              default_name: str) -> MyiaASTNode:
        if stash is not None:
            name, bindings = stash
            if isinstance(name, (Symbol, Tuple)):
                sym = name
            else:
                sym = self.gen.sym(name or default_name)
            bindings.append((sym, result))
            return sym
        return result

    def transform_arguments(self,
                            args: List[MyiaASTNode],
                            constructor: Callable,
                            stash: StashType,
                            base_name: str = None,
                            tags: Optional[List[str]] = None) \
            -> MyiaASTNode:

        bindings: Bindings = []
        new_args: List[Symbol] = []

        if base_name is None:
            fn, *args = args
            fn = self.transform(fn, stash=(None, bindings))
            assert isinstance(fn, Symbol)
            new_args.append(fn)
            base_label: Union[str, Symbol] = fn
            while isinstance(base_label, Symbol):
                base_label = base_label.label
            if "/" in base_label:
                base_name = ""
            elif base_label.startswith('#'):
                base_name = base_label
            else:
                base_name = re.split("[^A-Za-z_]", base_label)[0]

        if tags is None:
            tags = ['in' + str(i + 1) for i, _ in enumerate(args)]

        for arg, tag in zip(args, tags):
            if tag is False:
                new_arg = self.transform(arg)
            else:
                input_name = f'{base_name}/{tag}'
                new_arg = self.transform(arg, stash=(input_name, bindings))
            new_args.append(new_arg)

        app = constructor(*new_args)

        if stash is None:
            sym = self.stash((None, bindings), app, base_name + '/out')
            result = Let(bindings, sym)
        else:
            if bindings:
                result = Let(bindings, app)
            else:
                result = app

        return self.stash(stash, result, base_name + '/out')

    def transform_ApplyNode(self, node, stash=None) -> MyiaASTNode:
        return self.transform_arguments([node.fn] + node.args, Apply, stash)

    def transform_Symbol(self, node, stash=None) -> MyiaASTNode:
        return node

    def transform_ValueNode(self, node, stash=None) -> MyiaASTNode:
        return node

    def transform_LambdaNode(self, node, stash=None) -> MyiaASTNode:
        tr = ANormalTransformer(node.gen)
        result = create_lambda(node.ref, node.args, tr.transform(node.body),
                               node.gen, commit=False)
        return self.stash(stash, result, 'lambda')

    def transform_LetNode(self, node, stash=None) -> MyiaASTNode:
        new_let = []
        for s, b in node.bindings:
            bindings: List = []
            res = self.transform(b, stash=(s, bindings))
            assert len(bindings) <= 1
            if len(bindings) == 1:
                new_let.append(bindings[0])
            else:
                new_let.append((s, res))

        result = Let(new_let, self.transform(node.body))
        return self.stash(stash, result, 'let')

    def transform_TupleNode(self, node, stash=None) -> MyiaASTNode:
        def _Tuple(*args):
            return Tuple(args)
        return self.transform_arguments(node.values, _Tuple, stash, 'tup')

    def transform_ClosureNode(self, node, stash=None) -> MyiaASTNode:
        def rebuild(f, *args):
            return Closure(f, args)
        return self.transform_arguments([node.fn, *node.args],
                                        rebuild, stash, 'closure')

    def transform_BeginNode(self, node, stash=None) -> MyiaASTNode:
        stmts: List[MyiaASTNode] = \
            [stmt for stmt in node.stmts[:-1]
             if not isinstance(stmt, (Symbol, Value))] + node.stmts[-1:]
        if len(stmts) == 1:
            return self.transform(stmts[0], stash=stash)
        else:
            syms: List[LHS] = [nsym() for stmt in stmts]
            bindings = list(zip(syms, stmts))
            return self.transform_LetNode(Let(bindings, syms[-1]), stash)


class CollapseLet(Transformer):
    """
    Transform nested Lets into a single Let. That is to say:

        (let ((a (let (b c) x))
              (d y))
          (let ((e z))
            w))

    Becomes:

        (let ((b c)
              (a x)
              (d y)
              (e z))
          w)
    """
    __transform__ = 'collapse-let'

    def __init__(self) -> None:
        pass

    def transform_LetNode(self, node) -> MyiaASTNode:
        new_bindings: Bindings = []
        for s, b in node.bindings:
            b = self.transform(b)
            if isinstance(b, Let):
                new_bindings += b.bindings
                b = b.body
            new_bindings.append((s, b))
        body = self.transform(node.body)
        if isinstance(body, Let):
            return Let(new_bindings + body.bindings, body.body)
        return Let(new_bindings, body)

    def transform_Symbol(self, node) -> MyiaASTNode:
        return node

    def transform_ValueNode(self, node) -> MyiaASTNode:
        return node

    def transform_LambdaNode(self, node) -> MyiaASTNode:
        return create_lambda(node.ref, node.args, self.transform(node.body),
                             node.gen, commit=False)

    def transform_ApplyNode(self, node) -> MyiaASTNode:
        return Apply(self.transform(node.fn),
                     *[self.transform(a) for a in node.args])

    def transform_TupleNode(self, node) -> MyiaASTNode:
        return Tuple(self.transform(a) for a in node.values)

    def transform_ClosureNode(self, node) -> MyiaASTNode:
        return node
