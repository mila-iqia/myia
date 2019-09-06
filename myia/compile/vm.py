"""Implementation of a prototype optimized VM in python."""

from .. import xtype
from ..utils import TaggedValue


class struct_partial:
    """Representation for the result of a partial()."""

    def __init__(self, fn, args):
        """Initialize struct_partial."""
        self.fn = fn
        self.args = args

    def __repr__(self):  # pragma: no cover
        return f"partial({self.fn}, {self.args})"


class FinalVM:
    """Run a sequence of instructions.

    These instructions can represent multiple graphs with arbitrary
    recursion between them.
    """

    def __init__(self, code, backend):
        """Create a VM with the specified instructions."""
        self.code = tuple(code)
        self.stack = [None]  # The value stack
        self.retp = [-1]  # The call stack
        self.pc = 0  # program counter (next instruction)
        self.sp = 0  # stack pointer (for the value stack)
        self.backend = backend

    def _push(self, v):
        """Push a value to the stack."""
        self.stack[self.sp] = v
        self.sp += 1

    def _pop(self, n=1):
        """Pop a number of values off the stack returning the top.

        This also clears values that were popped off the stack.
        """
        v = self.stack[self.sp - 1]
        for p in range(n):
            self.stack[self.sp - p - 1] = None
        self.sp -= n
        return v

    def _move_stack(self, nitems, height):
        """Move a range of values down the stack, clearing the excess.

        This is used to implement tailcalls.
        """
        self.stack[self.sp - height:self.sp - (height - nitems)] = \
            self.stack[self.sp - nitems:self.sp]
        self._pop(height - nitems)

    def _ref(self, i):
        """Fetch a value from the stack."""
        return self.stack[self.sp + i]

    def _pushp(self):
        """Push the pc on the call stack (call)."""
        self.retp.append(self.pc)

    def _popp(self):
        """Restore the pc from the call stack (return)."""
        v = self.retp.pop()
        assert isinstance(v, int)
        self.pc = v

    def _do_jmp(self, jmp):
        """Jump to the specified position.

        This also handles jumping to a partial.
        """
        if isinstance(jmp, struct_partial):
            self.inst_pad_stack(len(jmp.args))
            for a in reversed(jmp.args):
                self._push(a)
            jmp = jmp.fn
        assert isinstance(jmp, int)
        self.pc = jmp

    def __call__(self, *args):
        """Shortcut to eval()."""
        return self.eval(args)

    def eval(self, args):
        """Evalute the code for this vm with the passed-in arguments."""
        # reset the runtime to initial values
        self.stack = [None] * len(args)  # The value stack
        self.retp = [-1]  # The call stack
        self.pc = 0  # program counter (next instruction)
        self.sp = 0  # stack pointer (for the value stack)

        # Calling convention is to push arguments from last to first
        # because it makes partial application easier.
        for a in reversed(args):
            self._push(a)

        # Main runtime loop
        while self.pc >= 0:
            instr = self.code[self.pc]
            impl = getattr(self, f'inst_{instr[0]}', None)
            if impl is None:
                raise AssertionError(f'Unknown instruction {instr[0]}')
            self.pc += 1
            impl(*instr[1:])

        # When we reach here there should be a single value on the
        # value stack and it is the return value for the evaluation.
        assert self.sp == 1, self.sp
        return self.stack[0]

    def inst_call(self, jmp):
        """Call.

        Will push the current pc on the call stack and jump to the
        given referrence.  Arguments are assumed to already be on the stack.

        Arguments:
            jmp: stack reference to a callable (code position or partial).

        """
        self._pushp()
        self._do_jmp(self._ref(jmp))

    def inst_tailcall(self, jmp, height, nargs):
        """Tail call.

        Will clear `height` values off the stack moving the last
        `nargs` ones down to serve as arguments for the calls, then
        jumps to the given reference.  This does not push the pc on
        the call stack.

        Arguments:
            jmp: stack reference to a callable (code position or partial).
            height: height of the stack relative to the previous
                    function (includes arguments)
            nargs: number of arguments passed to the called reference.

        """
        jmp = self._ref(jmp)
        self._move_stack(nargs, height)
        self._do_jmp(jmp)

    def inst_return(self, rpos, height):
        """Return.

        Clears the stack while ensuring that the specified value ends
        up at the top of the remaing stack.  Then jumps to the pc at
        the top of the call stack.

        Arguments:
            rpos: reference to the return value
            height: stack height to clear (includes arguments)

        """
        rv = self._ref(rpos)
        self._pop(height)
        self._push(rv)
        self._popp()

    def inst_partial(self, fn_, *args_):
        """Create a partial application.

        Bundles together the callable and the arguments and pushes
        that onto the stack.  The resulting value is callable and will
        transparently add the specified arguemnts at the beginning of
        any other arguments received.

        Arguments:
            fn_: callable reference
            args_: arguments references

        """
        fn = self._ref(fn_)
        assert not isinstance(fn, struct_partial), \
            ("You found a nested partial case, please report it "
             "so we can add it to the tests")
        args = tuple(self._ref(a) for a in args_)
        self._push(struct_partial(fn, args))

    def inst_switch(self, cond, vtrue, vfalse):
        """Switch.

        This will fetch the conditional and push either vtrue or
        vfalse depending on its value.

        Arguments:
            cond: boolean value
            vtrue: reference
            vfalse: reference

        """
        if self.backend.to_scalar(self._ref(cond)):
            self._push(self._ref(vtrue))
        else:
            self._push(self._ref(vfalse))

    def inst_tuple(self, *args):
        """Create a tuple from the given arguments and push it on the stack.

        Arguments:
           *args: tuple elements

        """
        self._push(tuple(self._ref(a) for a in args))

    def inst_bool_and(self, a, b):
        """Push the idx-th element from the list.

        Arguments:
           a: a boolean
           b: a boolean

        """
        a = self.backend.to_scalar(self._ref(a))
        b = self.backend.to_scalar(self._ref(b))
        self._push(self.backend.from_scalar(a and b, xtype.Bool))

    def inst_tuple_getitem(self, t, idx):
        """Get a item from a tuple.

        Arguments:
           t: tuple
           idx: index
        """
        a = self._ref(t)
        idx = self.backend.to_scalar(self._ref(idx))
        self._push(a[idx])

    def inst_tuple_setitem(self, t, idx, v):  # pragma: no cover
        """Set an item in a tuple.

        Arguments:
           t: tuple
           idx: index
           v: value to set
        """
        raise RuntimeError("Please report your test case so that "
                           "we can add it to the test suite")
        t = self._ref(t)
        idx = self.backend.to_scalar(self._ref(idx))
        v = self._ref(v)
        self._push(t[:idx] + (v,) + t[idx + 1:])

    def inst_tagged(self, x, tag):
        """Create a TaggedValue.

        Arguments:
           x: some object
           tag: the tag
        """
        x = self._ref(x)
        tag = self.backend.to_scalar(self._ref(tag))
        self._push(TaggedValue(tag, x))

    def inst_hastag(self, x, tag):
        """Check if a TaggedValue has the given tag.

        Arguments:
           x: TaggedValue
           tag: the tag
        """
        x = self._ref(x)
        tag = self.backend.to_scalar(self._ref(tag))
        res = self.backend.from_scalar(x.has(tag), xtype.Bool)
        self._push(res)

    def inst_casttag(self, x, tag):
        """Extract the value of a TaggedValue.

        Arguments:
           x: TaggedValue
           tag: the expected tag
        """
        x = self._ref(x)
        tag = self.backend.to_scalar(self._ref(tag))
        self._push(x.cast(tag))

    def inst_unsafe_static_cast(self, x, type):
        """No-op.

        Arguments:
           x: Value
           type: the type to static cast to (ignored)
        """
        self._push(self._ref(x))

    def inst_env_getitem(self, env, idx, default):
        """Get an item from a grad environment."""
        env = self._ref(env)
        if len(env) < idx + 1 or env[idx] is None:
            self._push(self._ref(default))
            return
        self._push(env[idx])

    def inst_env_setitem(self, env, idx, val):
        """Set an item in a grad environment."""
        env = self._ref(env)
        before = env[:idx] + (None,) * (idx - len(env))
        after = env[idx + 1:]
        self._push(before + (self._ref(val),) + after)

    def inst_push(self, v):
        """Push a value on the stack.

        Used to push constant values

        Arguments:
            v: value to push

        """
        self._push(v)

    def inst_dup(self, rpos):
        """Duplicate a value already on the stack.

        Arguments:
            rpos: stack reference

        """
        self._push(self._ref(rpos))

    def inst_pad_stack(self, sz):
        """Pad stack.

        This should be the first instruction in any function.  It
        allows the vm to ensure that the function will have the stack
        space it needs.

        Arguments:
            sz: stack space

        """
        need = sz - (len(self.stack) - self.sp)
        if need > 0:
            self.stack.extend([None] * need)

    def inst_external(self, fn, args):
        """Call external function.

        This will call the provided function with the specified values
        and push any outputs that function has (may be more than one).

        Arguments:
           fn: Callable external function.
           args: sequence of stack references.

        """
        outs = fn(*(self._ref(a) for a in args))
        for o in outs:
            self._push(o)
