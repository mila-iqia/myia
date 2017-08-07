from typing import Union, Any, List, cast, \
    Sequence, Tuple as TupleT, Optional, Callable

import re
from myia.ast import \
    MyiaASTNode, Apply, Symbol, Value, Let, Lambda, \
    If, Closure, Tuple, Transformer
from myia.front import GenSym


# TODO: Lambda should never be a sub-expression, it should be pulled
#       out to the top-level, and wrapped with Closure if needed.


def a_normal(node: MyiaASTNode) -> MyiaASTNode:
    """
    transform the expression represented by this node in
    A-normal form (ANF). ANF forbids expression nesting,
    so e.g. (f (g x)) would become:
    (let ((tmp (g x))
          (out (f tmp)))
      out)

    :param node: The expression to process
    :return: The expression in A-normal form
    """
    node = ANormalTransformer().transform(node)
    node = CollapseLet().transform(node)
    return node


BindingType = TupleT[Symbol, MyiaASTNode]
StashType = TupleT[Optional[str], List[BindingType]]


class ANormalTransformer(Transformer):
    def __init__(self, gen: GenSym = None) -> None:
        self.gen = gen

    def stash(self,
              stash: StashType,
              result: MyiaASTNode,
              default_name: str) -> MyiaASTNode:
        if stash is not None:
            name, bindings = stash
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

        bindings: List[BindingType] = []
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

    def transform_Apply(self, node, stash=None) -> MyiaASTNode:
        return self.transform_arguments([node.fn] + node.args, Apply, stash)

    def transform_Symbol(self, node, stash=None) -> MyiaASTNode:
        return node

    def transform_Value(self, node, stash=None) -> MyiaASTNode:
        return node

    def transform_Lambda(self, node, stash=None) -> MyiaASTNode:
        tr = ANormalTransformer(node.gen)
        result = Lambda(node.args,
                        tr.transform(node.body),
                        node.gen)
        result.ref = node.ref
        result.global_env = node.global_env
        return self.stash(stash, result, 'lambda')

    def transform_If(self, node, stash=None) -> MyiaASTNode:
        return self.transform_arguments([node.cond, node.t, node.f],
                                        If, stash, 'if',
                                        ['cond', 'then', 'else'])

    def transform_Let(self, node, stash=None) -> MyiaASTNode:
        result = Let([(s, self.transform(b)) for s, b in node.bindings],
                     self.transform(node.body))
        return self.stash(stash, result, 'if')

    def transform_Tuple(self, node, stash=None) -> MyiaASTNode:
        def _Tuple(*args):
            return Tuple(args)
        return self.transform_arguments(node.values, _Tuple, stash, 'tup')

    def transform_Closure(self, node, stash=None) -> MyiaASTNode:
        def rebuild(f, *args):
            return Closure(f, args)
        return self.transform_arguments([node.fn, *node.args],
                                        rebuild, stash, 'closure')

    def transform_Begin(self, node, stash=None) -> MyiaASTNode:
        stmts = [stmt for stmt in node.stmts[:-1]
                 if not isinstance(stmt, (Symbol, Value))] + node.stmts[-1:]
        if len(stmts) == 1:
            return self.transform(stmts[0], stash=stash)
        else:
            syms = [self.gen('_') for stmt in stmts]
            bindings = list(zip(syms, stmts))
            return self.transform_Let(Let(bindings, syms[-1]), stash)


class CollapseLet(Transformer):
    def __init__(self) -> None:
        pass

    def transform_Let(self, node) -> MyiaASTNode:
        new_bindings: List[BindingType] = []
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

    def transform_Value(self, node) -> MyiaASTNode:
        return node

    def transform_Lambda(self, node) -> MyiaASTNode:
        result = Lambda(node.args,
                        self.transform(node.body),
                        node.gen)
        result.ref = node.ref
        result.global_env = node.global_env
        return result

    def transform_Apply(self, node) -> MyiaASTNode:
        return Apply(self.transform(node.fn),
                     *[self.transform(a) for a in node.args])

    def transform_Tuple(self, node) -> MyiaASTNode:
        return Tuple(self.transform(a) for a in node.values)

    def transform_If(self, node) -> MyiaASTNode:
        return If(self.transform(node.cond),
                  self.transform(node.t),
                  self.transform(node.f))

    def transform_Closure(self, node) -> MyiaASTNode:
        return node
