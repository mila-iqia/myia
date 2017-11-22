
from ..stx import GenSym, ANORM, TMP, is_global, \
    Symbol, LetNode, LambdaNode, TupleNode, \
    ApplyNode, ValueNode, ClosureNode
from ..transform import a_normal
from .graph import IRNode, IRGraph
from ..symbols import builtins


gen = GenSym(':value')


def lambda_to_ir(lbda):

    assoc = {}

    def fetch(x):
        if x not in assoc:
            if is_global(x):
                assoc[x] = IRNode(None, x, x)
            elif isinstance(x, Symbol):
                assoc[x] = IRNode(g, x)
            elif isinstance(x, ValueNode):
                assoc[x] = IRNode(None, gen('value'), x.value)
            else:
                assoc[x] = IRNode(None, gen('value'), x)
        return assoc[x]

    lbda = a_normal(lbda)

    let = lbda.body

    ref = lbda.ref
    if ref.relation is ANORM:
        ref = ref.label

    g = IRGraph(None, ref, lbda.gen)
    g.inputs = tuple(fetch(sym) for sym in lbda.args)
    rval = IRNode(None, g.tag, g)

    if isinstance(let, Symbol):
        g.output = fetch(let)
        return rval
    elif isinstance(let, ValueNode):
        g.output = fetch(let)
        return rval
    else:
        g.output = fetch(let.body)
        assert isinstance(let, LetNode)

    def assign(k, v, idx=None):
        if isinstance(k, TupleNode):
            tmp = lbda.gen(TMP)
            assign(tmp, v, idx)
            for i, kk in enumerate(k.values):
                assign(kk, tmp, i)
            return

        wk = fetch(k)
        if idx is not None:
            wk.set_app(fetch(builtins.index), [fetch(v), fetch(idx)])
            return

        if isinstance(v, ApplyNode):
            args = [fetch(a) for a in v.args]
            wk.set_app(fetch(v.fn), args)
        elif isinstance(v, (Symbol, ValueNode)):
            wk.set_app(fetch(builtins.identity), [fetch(v)])
        elif isinstance(v, ClosureNode):
            args = [fetch(v.fn)] + [fetch(a) for a in v.args]
            wk.set_app(fetch(builtins.partial), args)
        elif isinstance(v, TupleNode):
            args = [fetch(a) for a in v.values]
            wk.set_app(fetch(builtins.mktuple), args)
        else:
            raise MyiaSyntaxError('Illegal ANF clause.', node=v)

    for k, v in let.bindings:
        assign(k, v)

    return rval
