from tvm import relay
from tvm.relay import ir_pass
import numpy as np


class _LiveSet(relay.expr_functor.ExprMutator):
    """Compute the live set of globals, we use
       ExprMutator here, but really want a visitor
       as we don't care about the resulting Expr.
    """

    def __init__(self, mod):
        self.live_set = set()
        self.mod = mod
        super().__init__()

    def visit_global_var(self, var):
        if var not in self.live_set:
            self.live_set.add(var)
            self.visit(self.mod[var])
        return var

    def visit_if(self, ite):
        return relay.If(
            self.visit(ite.cond),
            self.visit(ite.true_branch),
            self.visit(ite.false_branch))


def _live_from_main(mod):
    ls = _LiveSet(mod)
    ls.visit(mod[mod.entry_func])
    return ls.live_set


def _optimize_func(mod, func):
    ck_expr = ir_pass.infer_type(func, mod=mod)
    simp_expr = ir_pass.simplify_inference(ck_expr)
    ck_simp = ir_pass.infer_type(simp_expr, mod=mod)
    fused_expr = ir_pass.fuse_ops(ck_simp)
    ck_fused = ir_pass.infer_type(fused_expr, mod=mod)
    return ck_fused


def _placeholder_body(type):
    if isinstance(type, relay.TensorType):
        sh = [int(sh) for sh in type.shape]
        return relay.const(np.array(np.random.rand(*sh)).astype(type.dtype),
                           dtype=type.dtype)
    elif isinstance(type, relay.TupleType):
        return relay.Tuple([_placeholder_body(f) for f in type.fields])
    elif isinstance(type, relay.FuncType):
        params = []
        for arg_ty in type.arg_types:
            params.append(relay.var("p", type_annotation=arg_ty))

        return relay.Function(
            params,
            _placeholder_body(type.ret_type),
            ret_type=type.ret_type)


def build_module(funcs):
    """
    Workaround for type checker and mutually recursive functions.
    """
    mod = relay.Module({})
    for gv in funcs:
        func = funcs[gv]
        body = _placeholder_body(func.ret_type)
        mod[gv] = relay.Function(func.params, body, func.ret_type)

    for gv in funcs:
        mod[gv] = funcs[gv]

    return mod


def optimize(mod):
    """
    Modules are the only mutable piece of Relay.
    We write an optimization pass over the module
    which destructably updates each function while
    optimizing.
    """
    ls = _live_from_main(mod)
    for var in ls:
        mod[var] = _optimize_func(mod, mod[var])
