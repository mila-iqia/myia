"""
Defines a program transformation that can map any function to a
new function that can compute its own back-propagated gradient
while reusing intermediate results. This is the transformation
originally formulated in [1].

See the Gradient section in DEVELOPERS.md for information to
better understand the contents of this file.

[1] B. Pearlmutter, J. Siskind,
    Reverse-Mode AD in a Functional Framework:
    Lambda the Ultimate Backpropagator (2008)
    http://www.bcl.hamilton.ie/~barak/papers/toplas-reverse.pdf
"""

from typing import Dict, List, Tuple as TupleT, Any, \
    Union, cast, Optional, Sequence, Iterable, Callable, Set

from .stx import \
    LHS, Binding, Bindings, Transformer, GenSym, MyiaASTNode, \
    Symbol, Value, Lambda, Let, Apply, Tuple, Closure, maptup, \
    About, transformer_method, bsym, gsym, nsym, \
    JTAG, SENS, BPROP, BPROP_CLOS, NULLSYM, \
    TMP_LET, TMP_BPROP, TMP_SENS
from .interpret import \
    root_globals, evaluate, \
    PrimitiveImpl, FunctionImpl, ClosureImpl
from .symbols import builtins
from copy import copy
from .compile import a_normal
from .util import Props, Keyword, buche
from collections import OrderedDict
from .symbols import ZERO


LeafType = Union[Symbol, Value]


#################
# Helpers for J #
#################


def JGrad(x: FunctionImpl) -> Callable[[int], FunctionImpl]:
    """
    Helper function that creates a gradient factory for
    a FunctionImpl.

    See previous section on Partial Application for the
    purpose of the ``nargs_closure`` argument.
    """

    # We cache the compilation results.
    _cache: Dict[int, FunctionImpl] = {}

    def make_grad(nargs_closure: int) -> FunctionImpl:
        gfn = _cache.get(nargs_closure, None)
        if gfn:
            return gfn

        normalized = a_normal(x.ast)
        assert isinstance(normalized, Lambda)
        assert x.ast.ref

        # Generate the gradient expression
        G = Grad(
            name = x.ast.ref,
            primal = normalized,
            nargs_closure = nargs_closure
        )
        g = G.transform()

        # Create a FunctionImpl
        gfn = evaluate(g, G.global_env)

        # Don't forget to cache.
        _cache[nargs_closure] = gfn
        return gfn
    return make_grad


def JX(x: Union[PrimitiveImpl, FunctionImpl],
       nargs_closure: int) -> FunctionImpl:
    """
    Helper function for the gradient of PrimitiveImpl or
    FunctionImpl, given nargs_closure closure arguments.

    See previous section on Partial Application for the
    purpose of the ``nargs_closure`` argument.
    """
    if isinstance(x, PrimitiveImpl):
        # x.grad is set by the bprop_impl decorator. If it is
        # None, it means no one defined a gradient for that
        # operation.
        assert x.grad is not None
        return x.grad(nargs_closure)
    elif isinstance(x, FunctionImpl):
        if not x.grad:
            x.grad = JGrad(x)
        return x.grad(nargs_closure)
    else:
        raise TypeError(f'JX applied on wrong type: {x}')


########
# Grad #
########


class Grad:
    """
    Transform a Lambda into a Lambda that returns a backpropagator
    in addition to its normal return value.
    """

    def __init__(self,
                 name: Symbol,
                 primal: Lambda,
                 nargs_closure = 0) -> None:
        self.name = name
        assert isinstance(primal, Lambda)
        self.primal = primal
        self.gensym = primal.gen
        assert primal.global_env
        self.global_env = primal.global_env
        self.tagged_map: Dict[Symbol, Symbol] = {}
        self.sensitivity_map: Dict[Symbol, Symbol] = {}
        self.backpropagator_map: Dict[LHS, Symbol] = {}
        self.zeros: Bindings = []
        self.bprop_variables: Dict[Symbol, bool] = OrderedDict()
        self.nargs_closure = nargs_closure
        self.relevant: Set[Symbol] = None

    def get_relevant(self,
                     bindings: Bindings,
                     of_interest: List[Symbol]) -> Set[Symbol]:
        """
        Track which gradients are necessary to compute the gradients
        of the variables in of_interest. This will mostly prune the
        gradients of the functions we call.
        """
        deps: Dict[Symbol, Set[Symbol]] = {}
        for var, value in bindings:
            if isinstance(value, Apply):
                dependents = [value.fn] + value.args
            elif isinstance(value, Symbol):
                dependents = [value]
            elif isinstance(value, Tuple):
                dependents = value.values
            elif isinstance(value, Closure):
                dependents = [value.fn] + value.args
            else:
                dependents = []
            for var2 in dependents:
                if isinstance(var2, Symbol):
                    d = deps.setdefault(var2, set())
                    maptup(d.add, var)

        def acquire(sym):
            if sym in relevant:
                pass
            else:
                relevant.add(sym)
                for sym2 in deps.get(sym, set()):
                    acquire(sym2)

        relevant: Set[Symbol] = set()
        for sym in of_interest:
            acquire(sym)

        return relevant

    @transformer_method('g:phi', 2)
    def phi(self, var: LHS, value: MyiaASTNode) -> Bindings:
        """
        Given a variable and the expression it is bound to,
        return a list of (variable, value) bindings to append to
        the forward phase. See p. 26 in the P&S paper.
        """

        # Keep in mind:
        # self.tagged_var(x)          ==>  ↑x, x must be LHS
        # self.tagged_expr(x)         ==>  ↑x, or x if x is a Value
        # self.backpropagator_var(x)  ==>  ♢x

        if isinstance(value, Symbol):
            # Original:     x = y
            # Transformed:  ↑x = ↑y
            return [(self.tagged_var(var), self.tagged_expr(value))]

        elif isinstance(value, Value):
            # Original:     x = 5
            # Transformed:  ↑x = 5
            return [(self.tagged_var(var), value)]

        elif isinstance(value, Tuple):
            # Original:     x = (y, z)
            # Transformed:  ↑x = (↑y, ↑z)
            return [(self.tagged_var(var),
                     Tuple(self.tagged_expr(a) for a in value.values))]

        elif isinstance(value, Apply):
            # Original:     x = f(y)
            # Transformed:  ↑x, ♢x = ↑f(↑y)
            return [(Tuple([self.tagged_var(var),
                            self.backpropagator_var(var)]),
                     Apply(self.tagged_expr(value.fn),
                           *[self.tagged_expr(a) for a in value.args]))]

        elif isinstance(value, Closure):
            # Original:     x = Closure(f, y, z)
            # Transformed:  ↑f = JX(f, 2)  # evaluated immediately
            #               ↑x = Closure(↑f, ↑y, ↑z)
            # where the last argument to JX is the number of free
            # variables the function is referring to (y and z in
            # the example).

            # We should always statically know ``f`` in order to
            # apply this rule, but this will always be the case
            # if we have a closure in the code.
            assert isinstance(value.fn, Symbol)
            if value.fn.namespace not in {'global', 'builtin'}:
                raise Exception(
                    'First argument to Closure'
                    ' should always be a global variable.'
                )

            args = [self.tagged_expr(a) for a in value.args]
            fn = evaluate(value.fn, self.global_env)
            jfn = JX(fn, len(value.args))
            expr = Closure(jfn.ast.ref, args)

            return [(self.tagged_var(var), expr)]

        else:
            raise Exception(f'phi is not defined on node type: {value}')

    @transformer_method('g:rho', 2)
    def rho(self, var: LHS, value: MyiaASTNode) -> Bindings:
        """
        Given a variable and the expression it is bound to,
        return a list of (variable, value) bindings to prepend
        to the backward phase. See p. 26 in the P&S paper.
        """

        # Keep in mind:
        # self.sensitivity_var(x)     ==>  ∇x
        # self.backpropagator_var(x)  ==>  ♢x
        # x += y means x_2 = mapadd(x, y), where x_2 is a fresh
        # variable (to keep single assignment property)

        def args_cast(args: List[MyiaASTNode]) -> List[LeafType]:
            # Just a helper function to satisfy mypy
            assert all(isinstance(a, (Symbol, Value)) for a in args)
            return cast(List[LeafType], args)

        sen = self.sensitivity_value(var)
        if sen == Value(ZERO):
            # In this case we know ∇x's current value is 0.
            # Consequently, this term cannot possibly contribute
            # to any other sensitivity variable, so we save
            # ourselves the trouble.
            return []
        else:
            # If ``var`` is a Tuple, some values might be
            # ZERO and others not, and that can be a problem
            # because ZERO is not conformant (shape-wise).
            # Therefore, we must backtrack and get something
            # that we know is conformant.
            sen = self.conformant_sensitivity_value(var)

        if isinstance(value, Symbol):
            # Original:     x = y
            # Transformed:  ∇x += ∇y
            # return self.accum([value], Tuple([sen]))
            return self.accum_single(value, sen)

        elif isinstance(value, Value):
            # Original:     x = 5
            # Transformed:  <nothing to do>
            return []

        elif isinstance(value, Tuple):
            # Original:     x = (y, z)
            # Transformed:  ∇y, ∇z += ∇x
            args = args_cast(value.values)
            return self.accum_multi(args, sen)

        elif isinstance(value, Apply):
            # Original:     x = f(y)
            # Transformed:  ∇f, ∇y = ♢x(∇x)
            args = args_cast([value.fn, *value.args])
            bprop_var = self.backpropagator_var(var)
            increment = Apply(bprop_var, sen)
            self.bprop_variables[bprop_var] = True
            return self.accum_multi(args, increment)

        elif isinstance(value, Closure):
            # Original:     x = Closure(f, y, z)
            # Transformed:  ∇y, ∇z += ∇x
            # Why yes, this works the same as Tuple.
            args = args_cast(value.args)
            return self.accum_multi(args, sen)

        else:
            raise Exception(f'rho is not defined on node type: {value}')

    def accum_multi(self, vars: List[LeafType], value: MyiaASTNode) \
            -> Bindings:
        """
        Return code to accumulate the gradients returned as ``value``
        into a tuple of ``vars``. The result will be one of:

            A) (∇v1_2, ∇v2_2, ...) = mapadd((∇v1, ∇v2, ...), value)
            B) (∇v1, ∇v2, ...) = value
            C) (∇v1_2, tmp, ...) = mapadd((∇v1, ∇v1, ...), value)
               ∇v1_3 = mapadd(∇v1_2, tmp)
               ...

        A is the "normal" case. The following special conditions are
        handled by ``accum_multi``:

        * If ``v`` is a Value, then we accumulate into a dummy variable.
          This will happen with e.g.: ``x = y * 2``
        * If the gradient of ``v`` is not useful to calculate the gradients
          we want to return, we also generate a dummy variable.
        * Every ``var`` which is a Value or has no sensitivity value
          at the moment will be represented as ZERO in the first argument
          to ``mapadd``. This will happen if this is the very first
          gradient contribution for these variables.
        * If every ``var`` is ZERO, there is no need for ``mapadd`` and
          we can optimize to case B.
        * If two or more ``vars`` are the *same variable*, then we create
          temporary variables to hold the extra contributions and we
          append ``mapadd`` statements to merge them. This is case C.
          If we don't do that, we lose contributions. It's not an edge
          case, it's quite common, so this is very important to get right.
          This will happen with e.g.: ``x = y * y``
        """

        # Track which variables we have already seen (check for case C)
        seen: Set[Symbol] = set()
        # Accumulate the variables on the left of =
        lhs_vars: List[Symbol] = []
        # Accumulate the variables for the first argument of mapadd
        rhs_vars: List[MyiaASTNode] = []
        # Accumulate bindings
        bindings: Bindings = []

        for var in vars:
            with About(var, 'g:sens_acc'):
                if isinstance(var, Value) or var not in self.relevant:
                    # Dummies
                    lhs_vars.append(nsym())
                    rhs_vars.append(Value(ZERO))
                elif var in seen:
                    # We have a duplicate variable, so we make a temp
                    g = self.gensym(var, TMP_SENS)
                    lhs_vars.append(g)
                    # We must add the temp's value to the sensitivity
                    # variable for var.
                    app = Apply(builtins.mapadd,
                                g,
                                self.conformant_sensitivity_value(var))
                    lhs = self.new_sensitivity_var(var)
                    bindings.append((lhs, app))
                    # We make a dummy zero for mapadd's first argument,
                    # so we're only getting the contribution for the
                    # argument into the temp (we wouldn't want to count
                    # its previous value more than once)
                    rhs_vars.append(Value(ZERO))
                else:
                    rhs = self.sensitivity_value(var)
                    lhs_vars.append(self.new_sensitivity_var(var))
                    seen.add(var)
                    rhs_vars.append(rhs)

        new_value: MyiaASTNode
        with About(value, 'g:sens_rhs'):
            if all(x == Value(ZERO) for x in rhs_vars):
                new_value = value
            else:
                new_value = Apply(builtins.mapadd, Tuple(rhs_vars), value)

        # We must prepend the main operation to the extra bindings
        # we created for the duplicates
        binding: Binding = (Tuple(lhs_vars), new_value)
        return [binding] + bindings

    @transformer_method('g:sens_acc', 2)
    def accum_single(self, v: LeafType, value) -> Bindings:
        if isinstance(v, Value):
            # No accumulation in non-variables.
            return []
        sen = self.sensitivity_value(v)
        new_sen = self.new_sensitivity_var(v)
        if sen == Value(ZERO):
            return [(new_sen, value)]
        else:
            return [(new_sen, Apply(builtins.mapadd, sen, value))]

    @transformer_method('g:tag')
    def tagged_var(self, v: LHS) -> LHS:
        """
        Return ``↑v``. Creates it if it does not exist.
        """
        if isinstance(v, Symbol):
            assert v.namespace not in {'global', 'builtin'}
            return copy(self.tagged_map.setdefault(v, self.gensym(v, JTAG)))
        elif isinstance(v, Tuple):
            rval = maptup(self.tagged_var, v)
            return rval
        else:
            raise TypeError(f'Cannot tag {v} of type {type(v)}')

    @transformer_method('g:tag')
    def tagged_expr(self, v: MyiaASTNode) -> MyiaASTNode:
        """
        * If ``v`` is a Value, return ``v``.
        * If ``v`` is a global Symbol, return ``J(v)``.
        * Otherwise return ``↑v``.
        """
        assert isinstance(v, (Symbol, Value))
        if isinstance(v, Value):
            return v
        if v.namespace in {'global', 'builtin'}:
            return Apply(builtins.J, v)
        else:
            return self.tagged_var(v)

    @transformer_method('g:sens')
    def sensitivity_value(self, v: LHS) -> MyiaASTNode:
        """
        Returns ``∇v``, the current sensitivity for the variable ``v``.
        If ``v`` was not set before, the return value will be
        ``ZERO``, otherwise it will be its current sensitivity
        variable.

        ``ZERO`` is not conformant with ``v``, which can be a problem.
        Therefore, only use ``sensitivity_value`` to get a suitable
        argument for the *first* argument to ``mapadd``. This is ok because
        we know that the second will be conformant and has the appropriate
        structure. Use ``conformant_sensitivity_value`` anywhere else.
        """
        if isinstance(v, Symbol):
            try:
                return copy(self.sensitivity_map[v])
            except KeyError:
                return Value(ZERO)
        else:
            rval = maptup(self.sensitivity_value, v)
            if all(v == Value(ZERO) for v in rval.values):
                # If all sensitivity values in this tuple are ZERO,
                # we can summarize the whole tuple with ZERO.
                return Value(ZERO)
            return rval

    @transformer_method('g:zinit')
    def zero_init(self, var: Symbol) -> Symbol:
        """
        Handle zero initialization code for a variable's gradient.
        That code is:

            ∇x = zeros_like(Jinv(↑x))

        ``Jinv(↑x)`` is the same as ``x``, but we don't have access to
        ``x`` since a transformed function ``↑f`` receives ``↑x``
        directly as an argument. Thankfully, the transformation is
        invertible, and that is why we use ``Jinv``.

        The initialization code is stored in ``self.zeros``, and ``∇x``
        is returned.
        """
        new_var = self.new_sensitivity_var(var)
        tagged = self.tagged_var(var)
        assert isinstance(tagged, Symbol)
        init = (new_var,
                Apply(builtins.zeros_like,
                      Apply(builtins.Jinv, tagged)))
        self.zeros.append(init)
        self.bprop_variables[tagged] = True
        return new_var

    @transformer_method('g:sens')
    def conformant_sensitivity_value(self, v: LHS) -> MyiaASTNode:
        """
        Return ``∇v`` if it already exists. If it does not, create it
        and initialize it with ``zero_init``. This differs from
        ``sensitivity_value`` in one important way, which is that
        ``sensitivity_value`` returns the ``ZERO`` placeholder if it
        does not find ``∇v``, whereas this creates a zero that has the
        same shape as ``v``.
        """
        if isinstance(v, Symbol):
            try:
                return copy(self.sensitivity_map[v])
            except KeyError:
                return self.zero_init(v)
        else:
            return maptup(self.conformant_sensitivity_value, v)

    @transformer_method('g:sens')
    def new_sensitivity_var(self, v: Symbol) -> Symbol:
        """
        Create a new sensitivity variable for v. This is used to preserve
        the single-assignment property: instead of ∇v = ∇v + x,
        we do ∇v_2 = ∇v + x. self.sensitivity_var maps to the latest
        return value for this function.
        """
        if isinstance(v, Symbol):
            new_v = self.gensym(v, SENS)
        else:
            raise TypeError(f'Cannot make sensitivity var for {v}')
        self.sensitivity_map[v] = new_v
        return new_v

    @transformer_method('g:bprop')
    def backpropagator_var(self, v: LHS) -> Symbol:
        """
        Return ``♢v``. Create it if it does not exist.
        """
        if isinstance(v, Tuple):
            # If we have a deconstructing assignment, we still
            # only get one backpropagator, so we create a new
            # variable to hold it.
            # TODO: try to derive a readable name from the tuple?
            sym = self.gensym(self.gensym(TMP_BPROP), BPROP_CLOS)
        else:
            sym = self.gensym(v, BPROP_CLOS)
        return copy(self.backpropagator_map.setdefault(v, sym))

    def transform(self) -> Symbol:
        """
        Perform the code transform on self.primal.
        """

        # The arguments of the function. We want the gradients of
        # these.
        args = self.primal.args

        # The body of the function, which we need to be a Let in
        # a-normal form.
        let = self.primal.body
        if isinstance(let, Symbol):
            tmp = self.gensym(TMP_LET)
            let = Let([(tmp, let)], tmp)
        assert isinstance(let, Let)

        self.relevant = self.get_relevant(let.bindings, args)

        # We start by creating the sensitivity variable ``∇out``
        # which is the input to the backpropagator.
        assert isinstance(let.body, Symbol)
        out_sen = self.new_sensitivity_var(let.body)

        # Repeatedly call phi to build the forward pass.
        forward: Bindings = []
        for s, v in let.bindings:
            forward += self.phi(s, v)

        # Repeatedly call rho to build the backprop pass.
        backward: Bindings = []
        for s, v in reversed(let.bindings):
            backward += self.rho(s, v)

        ###############################
        # Build the backpropagator ♦f #
        ###############################

        # We return the sensitivity variables for all of the inputs.
        backp_all_ret = [self.conformant_sensitivity_value(arg)
                         for arg in args]
        # The return value of ♦f: we group together the first
        # nargs_closure gradients because they are the closure's
        # gradient.
        backp_ret = Tuple([
            Tuple(backp_all_ret[:self.nargs_closure]),
            *backp_all_ret[self.nargs_closure:]
        ])
        # Calls to rho had the side effect of populating bprop_variables
        # with all the variables we need for the bprop. We will need
        # to feed them in as arguments to ♦f.
        backp_args = list(self.bprop_variables.keys())
        # TODO: copying these args is kind of iffy. They should be
        # different versions of the symbols, or use a different
        # namespace.
        backp_args_copy: Iterable[Symbol] = map(copy, backp_args)
        # ♦f
        backp_sym = self.global_env.gen(self.name, BPROP)
        backp_fn = Lambda([*backp_args_copy, out_sen],
                          Let(self.zeros + backward, backp_ret),
                          self.gensym)
        backp_fn.global_env = self.global_env
        backp_fn.ref = backp_sym
        self.global_env[backp_sym] = backp_fn
        root_globals[backp_sym] = backp_fn  # TODO: obviate

        ########################
        # Build the closure ♢f #
        ########################

        # First we will make a closure over ♦f and call it ♢f
        backp_cl = Closure(backp_sym, backp_args)
        backp_clsym = self.gensym(self.name, BPROP_CLOS)
        # We append the closure binding to forward
        forward.append((backp_clsym, backp_cl))

        ###################################
        # Build the augmented function ↑f #
        ###################################

        # ↑f returns (↑out, ♢f)
        augm_ret = Tuple([self.tagged_expr(let.body), backp_clsym])
        augm_body = Let(forward, augm_ret)
        # ↑f takes ↑ versions of f's arguments
        augm_args = list(map(self.tagged_var, args))

        # ↑f
        assert all(isinstance(arg, Symbol) for arg in augm_args)
        augm_fn = Lambda(cast(List[Symbol], augm_args),
                         augm_body, self.gensym)
        augm_sym = self.global_env.gen(self.name, JTAG)
        augm_fn.global_env = self.global_env
        augm_fn.ref = augm_sym
        self.global_env[augm_sym] = augm_fn
        root_globals[augm_sym] = augm_fn  # TODO: obviate

        # Set the primal field to the original function's symbol
        augm_fn.primal = self.name

        return augm_sym
