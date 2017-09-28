
import re
from enum import Enum, auto
from ..stx.nodes import Symbol
from ..lib import Primitive
from ..symbols import builtins


class BreakpointMode(Enum):
    FORWARD = auto()
    BACKWARD = auto()


class Breakpoint:
    def __init__(self, mode=BreakpointMode.FORWARD):
        self.mode = mode


class VMPrinter:
    def __init__(self, vm):
        self.vm = vm

    def __hrepr__(self, H, hrepr):
        res = H.tabbedView()
        frs = [(fr, -1) for fr in self.vm.frames]
        frs.append((self.vm.frame, 0))
        for fr, offset in frs:
            foc = fr.rel_node(offset)
            if foc:
                foc.annotations.add('error')
            c = fr.code
            l = c.lbda
            tab = H.tab(hrepr(l.ref))
            pane = H.pane(hrepr(l or c.node))
            v = H.view(tab, pane)(active=offset == 0)
            res = res(v)
            if foc:
                foc.annotations.remove('error')
        return res


def parse_command_specs(specs):
    command_map = {}
    for spec in specs:
        main = None
        for x in spec.split(';'):
            if ':' in x:
                pfx, sfx = x.split(':')
                full = x.replace(':', '')
                if main is None:
                    main = full
                for i in range(len(sfx) + 1):
                    command_map[pfx + sfx[:i]] = main
            else:
                if main is None:
                    main = x
                command_map[x] = main
    return command_map


class DebugController:
    __commands__ = parse_command_specs([
        ':step',
        'n:ext',
        'c:ontinue',
        'u:p',
        'd:own',
        'ex:pression;x;?',
        'vars;?*',
        't:op'
    ])

    def __init__(self, buche, reader, next_breakpoint=True):
        self.dbf = buche.open_slides('frame')
        self.db = buche.open_log('debug', force=True, hasInput=True)
        self.db.log('-- Debug --')
        self.next_breakpoint = next_breakpoint
        self.reader = reader

    async def command_step(self, vm, *args):
        self.next_breakpoint = True
        return True

    async def command_next(self, vm, *args):
        self.next_breakpoint = vm.frame
        return True

    async def command_continue(self, vm, *args):
        self.next_breakpoint = False
        return True

    async def command_up(self, vm, *args):
        self.db.log('TODO: UP FRAME', kind='error')

    async def command_down(self, vm, *args):
        self.db.log('TODO: DOWN FRAME', kind='error')

    async def command_top(self, vm, n):
        if n.strip() == '':
            n = 1
        else:
            n = int(n)
        if n == 1:
            self.db(vm.frame.top())
        else:
            self.db(vm.frame.stack[-n:])

    async def command_expression(self, vm, expr):
        fr = vm.frame
        all_results = {}
        for vname in expr.split():
            results = {}
            d = dict(fr.eval_env)
            d.update(fr.local_env)
            for sym, value in d.items():
                s = sym
                while isinstance(s, Symbol):
                    s = s.label
                if s == vname:
                    results[sym] = value
            if not results:
                self.db.log(f'Could not find a variable named `{vname}`',
                            kind='error')
            all_results.update(results)
        self.db.show(all_results, kind='result')

    async def wait_for_command(self, vm):
        message = await self.reader.read_async()
        ctype = message.command
        if ctype == 'input':
            cmd, *rest = re.findall(r'\w+|\W+',
                                    message.contents.strip() or ' ')
            cmd = cmd.strip()
            arg = ''.join(rest)
            canon = self.__commands__.get(cmd, None)
            self.db.log(message.contents or canon, kind='echo')
            if canon is None:
                self.db.log(f'Unknown command: {cmd}', kind='error')
            else:
                method = getattr(self, f'command_{canon}')
                if await method(vm, arg):
                    return
        return await self.wait_for_command(vm)

    def ignore_operation(self, oper):
        if isinstance(oper, Primitive):
            sym = oper.name
            if sym in {builtins.switch, builtins.mktuple}:
                return True
        return False

    async def __call__(self, vm):
        instr = vm.frame.current_instruction()
        # self.db(instr)
        if isinstance(vm.frame.top(), Breakpoint):
            self.next_breakpoint = True

        if not instr or instr.command != 'reduce':
            return
        if self.ignore_operation(vm.frame.stack[-(instr.args[0] + 1)]):
            return

        if self.next_breakpoint is True or \
                self.next_breakpoint is vm.frame or \
                self.next_breakpoint is vm.frame.focus:
            pass
        else:
            return

        foc = vm.frame.focus
        if foc:
            self.dbf(VMPrinter(vm))
        else:
            self.db('No focus.')
        await self.wait_for_command(vm)
