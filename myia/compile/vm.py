class struct_partial:
    def __init__(self, fn, args):
        self.fn = fn
        self.args = args

    def __repr__(self):
        return f"partial({self.fn}, {self.args})"


class FinalVM:
    def __init__(self, code):
        self.stack = []
        self.retp = [-1]
        self.pc = 0
        self.sp = 0
        self.code = tuple(code)
        self.running = False

    def _push(self, v):
        self.stack[self.sp] = v
        self.sp += 1

    def _pop(self, n=1):
        v = self.stack[self.sp - 1]
        for p in range(n):
            self.stack[self.sp - p - 1] = None
        self.sp -= n
        return v

    def _move_stack(self, nitems, height):
        self.stack[self.sp - height:self.sp - (height - nitems)] = \
            self.stack[self.sp - nitems:self.sp]
        self._pop(height - nitems)

    def _ref(self, i):
        return self.stack[self.sp + i]

    def _pushp(self):
        self.retp.append(self.pc)

    def _popp(self):
        self.pc = self.retp.pop()

    def _do_jmp(self, jmp):
        print(f"jumping to {jmp}")
        print(f"stack = {self.stack} {self.sp}")
        while isinstance(jmp, struct_partial):
            print(f"is partial to {jmp.fn}, appending args")
            self.inst_pad_stack(len(jmp.args))
            for a in reversed(jmp.args):
                self._push(a)
            print(f"stack = {self.stack} {self.sp}")
            jmp = jmp.fn
        self.pc = jmp

    def eval(self, args):
        self.stack = [None] * len(args)
        self.retp = [-1]
        self.pc = 0
        self.sp = 0
        for a in reversed(args):
            self._push(a)

        self.running = True
        print("==== Start ====")
        print("Stack:")
        print(self.stack)
        print(f"pc = {self.pc}")
        print(self.code)
        print("=== Runtime ===")
        while self.pc >= 0:
            print(f"pc at {self.pc}")
            print(self.stack, self.sp)
            print(self.retp)
            instr = self.code[self.pc]
            print(f"instr = {instr[0]}")
            impl = getattr(self, f'inst_{instr[0]}', None)
            if impl is None:
                raise AssertionError(f'Unknown instruction {instr[0]}')
            self.pc += 1
            impl(*instr[1:])

        assert self.sp == 1, self.sp
        return self.stack[0]

    def inst_call(self, jmp):
        print(f"running call({jmp})")
        self._pushp()
        self._do_jmp(self._ref(jmp))

    def inst_tailcall(self, jmp, height, nargs):
        print(f"running tailcall({jmp}, {height}, {nargs})")
        jmp = self._ref(jmp)
        self._move_stack(nargs, height)
        self._do_jmp(jmp)

    def inst_return(self, rpos, height):
        print(f"running return({rpos}, {height})")
        rv = self._ref(rpos)
        self._pop(height)
        self._popp()
        self._push(rv)

    def inst_partial(self, fn_, *args_):
        fn = self._ref(fn_)
        args = tuple(self._ref(a) for a in args_)
        self._push(struct_partial(fn, args))

    def inst_if(self, cond, ftrue, ffalse):
        print(f"running if({cond}, {ftrue}, {ffalse})")
        if self._ref(cond):
            self.inst_call(ftrue)
        else:
            self.inst_call(ffalse)

    def inst_tailif(self, cond, ftrue, ffalse, height):
        print(f"running tailif({cond}, {ftrue}, {ffalse}, {height})")
        if self._ref(cond):
            self.inst_tailcall(ftrue, height, 0)
        else:
            self.inst_tailcall(ffalse, height, 0)

    def inst_push(self, v):
        print(f"running push({v})")
        self._push(v)

    def inst_dup(self, rpos):
        print(f"running dup({rpos})")
        self._push(self._ref(rpos))

    def inst_pop(self):
        print("running pop")
        self._pop()

    def inst_pad_stack(self, sz):
        print(f"running pad_stack({sz})")
        need = sz - (len(self.stack) - self.sp)
        if need > 0:
            self.stack.extend([None] * need)

    def inst_nnvm_call(self, mod, args):
        print(f"running nnvm_call({mod}, {args})")
        outs = mod(*(self._ref(a) for a in args))
        assert len(outs) == 1
        self._push(outs[0])
