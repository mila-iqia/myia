import re
from myia.ast import Apply, Let, Lambda, If, Tuple
from myia.front import GenSym
from uuid import uuid4 as uuid


def a_normal(node):
    """
    Convert the expression represented by this node in
    A-normal form (ANF). ANF forbids expression nesting,
    so e.g. (f (g x)) would become (let ((tmp (g x))) (f tmp))

    :param node: The expression to process
    :return: The expression in A-normal form
    """
    node = ANormalConverter(GenSym(uuid())).convert(node)
    node = CollapseLet().convert(node)
    return node


class ANormalConverter:
    def __init__(self, gen):
        self.gen = gen

    def convert(self, node, stash=False):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'convert_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type for a-normal conversion: {}".format(cls)
            )
        rval = method(node, stash)
        return rval

    def stash(self, stash, result, default_name):
        if stash is not False:
            name, stash = stash
            sym = self.gen.sym(name or default_name)
            stash.append((sym, result))
            return sym
        return result

    def convert_arguments(self, args, constructor,
                          stash, base_name=None, tags=None):

        bindings = []
        new_args = []

        if base_name is None:
            fn, *args = args
            fn = self.convert(fn, stash=(None, bindings))
            new_args.append(fn)
            if "/" in fn.label:
                base_name = ""
            else:
                base_name = re.split("[^A-Za-z_]", fn.label)[0]

        if tags is None:
            tags = ['in' + str(i + 1) for i, _ in enumerate(args)]

        for arg, tag in zip(args, tags):
            if tag is False:
                new_arg = self.convert(arg)
            else:
                new_arg = self.convert(arg, stash=(base_name + '/' + tag, bindings))
            new_args.append(new_arg)

        app = constructor(*new_args)
        if bindings:
            result = Let(bindings, app)
        else:
            result = app
        return self.stash(stash, result, base_name + '/out')


    def convert_Apply(self, node, stash=False):
        return self.convert_arguments((node.fn,) + node.args, Apply, stash)

    def convert_Symbol(self, node, stash=False):
        return node

    def convert_Value(self, node, stash=False):
        return node

    def convert_Lambda(self, node, stash=False):
        result = Lambda('#lambda', node.args, self.convert(node.body))
        return self.stash(stash, result, 'lambda')

    def convert_If(self, node, stash=False):
        return self.convert_arguments((node.cond, node.t, node.f),
                                      If, stash, 'if', ('cond', False, False))

    def convert_Let(self, node, stash=False):
        result = Let([(s, self.convert(b)) for s, b in node.bindings],
                     self.convert(node.body))
        return self.stash(stash, result, 'if')

    def convert_Tuple(self, node, stash=False):
        def _Tuple(*args): return Tuple(args)
        return self.convert_arguments(node.values, _Tuple, stash, 'tup')


class CollapseLet:
    def __init__(self):
        pass

    def convert(self, node):
        cls = node.__class__.__name__
        try:
            method = getattr(self, 'convert_' + cls)
        except AttributeError:
            raise Exception(
                "Unrecognized node type for let collapse: {}".format(cls)
            )
        rval = method(node)
        return rval

    def convert_Let(self, node):
        new_bindings = []
        for s, b in node.bindings:
            b = self.convert(b)
            if isinstance(b, Let):
                new_bindings += b.bindings
                b = b.body
            new_bindings.append((s, b))
        body = self.convert(node.body)
        if isinstance(body, Let):
            return Let(new_bindings + body.bindings, body.body)
        return Let(new_bindings, body)

    def convert_Symbol(self, node):
        return node

    def convert_Value(self, node):
        return node

    def convert_Lambda(self, node):
        return Lambda('#lambda', node.args, self.convert(node.body))

    def convert_Apply(self, node):
        return Apply(self.convert(node.fn), *[self.convert(a) for a in node.args])

    def convert_Tuple(self, node):
        return Tuple(self.convert(a) for a in node.values)

    def convert_If(self, node):
        return If(self.convert(node.cond), self.convert(node.t), self.convert(node.f))

