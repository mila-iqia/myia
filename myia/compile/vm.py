from ..prim.ops import if_, return_, add
from ..ir import Apply, toposort, is_parameter, is_apply, is_constant_graph, is_constant, Parameter, manage
from ..prim import Primitive


def cut(node):
    if is_apply(node):
        fn = node.inputs[0]
        if not is_constant(fn, Primitive):
            return True
        elif fn.value in (if_, return_):
            return True
    return False


def split_graph(graph):
    splits = []
    split = []

    for node in toposort(graph.return_):
        if cut(node):
            if len(split) != 0:
                splits.append(split)
            splits.append(node)
            split = []
        elif not (is_constant(node) or is_parameter(node)):
            split.append(node)

    return splits


import nnvm.compiler
import nnvm.symbol as sym


def counter():
    val = -1
    def next():
        nonlocal val
        val += 1
        return val
    return next


def nnvm_convert(lst):
    """Returns graph, inputs, output, constants"""
    eqv = {}
    inputs = []
    constants = []
    c = counter()

    def ref(n):
        if is_constant(n):
            name = f"cst{c()}"
            constants.append((name, n.value))
            eqv[n] = sym.Variable(name)
        elif n not in eqv:
            inputs.append(n)
            eqv[n] = sym.Variable(f"i{c()}")

        return eqv[n]

    for n in lst:
       assert is_apply(n)
       assert is_constant(n.inputs[0], Primitive)
       fn = n.inputs[0].value
       args = [ref(a) for a in n.inputs[1:]]
       if fn == add:
           eqv[n] = sym.elemwise_add(*args)
       else:
           raise NotImplementedError(fn)

    # TODO: figure out multiple outputs
    out = lst[-1]
    g = nnvm.graph.create(eqv[out])
    return g, inputs, constants, [out]
                                        

def convert_graph(graph):
    splits = split_graph(graph)

    height = 0
    max_height = 0
    slots = {}
    instrs = []

    def push(n):
        nonlocal height, max_height
        if n is not None:
            assert n not in slots
            slots[n] = height
        height += 1
        max_height = max(height, max_height)
        print(f"push height up {n}")
        return height

    def ref(n):
        if n not in slots and is_constant(n):
            if is_constant_graph(n):
                instrs.append(('push_graph', n.value))
            else:
                instrs.append(('push', n.value))
            push(n)
        return slots[n] - height

    def dup(n):
        nonlocal height, max_height
        i = ref(n)
        instrs.append(('dup', i))
        height += 1
        print(f"dup height up {n}")
        max_height = max(height, max_height)

    def ret(nargs):
        nonlocal height
        height -= nargs

    for p in graph.parameters:
        push(p)

    param_height = height
    # The pc
    height += 1
    max_height = max(height, max_height)
    print(f"(init) height = {height}, {max_height}")

    for split in splits:
        print(split)
        if isinstance(split, list):
            g, inputs, constants, outs = nnvm_convert(split)
            args = [ref(i) for i in inputs]
            instrs.append(('nnvm_call', g, args, constants))
            for o in outs:
                push(o)
        else:
            assert isinstance(split, Apply)
            fn = split.inputs[0]
            
            if is_constant(fn, Primitive):
                if fn.value == if_:
                    instrs.append(('if', ref(split.inputs[1]),
                                   split.inputs[2], split.inputs[3]))
                elif fn.value == return_:
                    print(f"(ret ) height = {height}, {max_height}")
                    instrs.append(('return', ref(split.inputs[1]),
                                   height - param_height - 1, param_height))
                    # To avoid pushing the split
                    continue
                else:
                    raise AssertionError("should not happen")
            elif is_constant_graph(fn):
                for i in split.inputs[1:]:
                    dup(i)
                instrs.append(('call_graph', fn))
                push(None) # The pc
                ret(len(split.inputs))
            else:
                for i in split.inputs[1:]:
                    dup(i)
                instrs.append(('call_ind', ref(fn)))
                push(None) # The pc
                ret(len(split.inputs))

            push(split)

    print(f"(end ) height = {height}, {max_height}")
    instrs.insert(0, ('pad_stack', max_height - (param_height + 1)))
    return instrs


def convert_and_link(graph):
    mng = manage(graph, weak=True)
    graphs = mng.graphs
    mapping = {}
    instrs = []

    # compile (aka convert)
    mapping[graph] = len(instrs)
    instrs.extend(convert_graph(graph))

    for g in (graphs - set([graph])):
        mapping[g] = len(instrs)
        instrs.extend(convert_graph(g))

    # link
    for i in range(len(instrs)):
        instr = instrs[i]
        if instr[0] == 'call_graph':
            if is_constant_graph(instr[1]):
                instrs[i] = ('call', mapping[instr[1].value])
        elif instr[0] == 'push_graph':
            instrs[i] = ('push', mapping[instr[1]])
        elif instr[0] == 'if':
            instrs[i] = ('if', instr[1], mapping[instr[2].value],
                         mapping[instr[3].value])

    return instrs


def stack_optimize(instrs):
    pass


class FinalVM:
    def __init__(self, code):
        self.stack = []
        self.pc = -1
        self.sp = 0
        self.code = tuple(code)
        self.running = False

    def _push(self, v):
        self.stack[self.sp] = v
        self.sp += 1

    def _pop(self, n=1):
        v = self.stack[self.sp - 1]
        self.sp -= n
        return v

    def _ref(self, i):
        return self.stack[self.sp + i]

    def eval(self, args):
        self.stack = [None] * (len(args) + 1)
        self.pc = -1
        self.sp = 0
        for a in args:
            self._push(a)
        self.inst_call(0)

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
            instr = self.code[self.pc]
            print(f"instr = {instr[0]}")
            impl = getattr(self, f'inst_{instr[0]}', None)
            if impl is None:
                raise AssertionError(f'Unknown instruction {instr[0]}')
            self.pc += 1
            impl(*instr[1:])

        assert self.sp == 1
        return self.stack[0]

    def inst_call(self, jmp):
        print(f"running call({jmp})")
        self._push(self.pc)
        self.pc = jmp

    def inst_call_ind(self, cpos):
        print(f"running call_ind({cpos})")
        jmp = self._ref(cpos)
        self.inst_call(jmp)

    def inst_return(self, rpos, height, nargs):
        print(f"running return({rpos}, {height}, {nargs})")
        rv = self._ref(rpos)
        self._pop(height)
        self.pc = self._pop()
        self._pop(nargs)
        self._push(rv)

    def inst_if(self, cond, ftrue, ffalse):
        print(f"running if({cond}, {ftrue}, {ffalse})")
        if self._ref(cond):
            self.inst_call(ftrue)
        else:
            self.inst_call(ffalse)

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
        self.stack.extend([None] * sz)
