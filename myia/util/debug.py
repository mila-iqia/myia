
import re
import os
from enum import Enum, auto
from ..stx.nodes import Symbol
from ..lib import Primitive
from ..symbols import builtins


_template_path = f'{os.path.dirname(__file__)}/debug-template.html'


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
                    command_map[pfx + sfx[:i]] = (main, spec)
            else:
                if main is None:
                    main = x
                command_map[x] = (main, spec)
    return command_map


class DebugController:
    __commands__ = parse_command_specs([
        ':step',
        'n:ext',
        'c:ontinue',
        'u:p',
        'd:own',
        'v:ar;?',
        't:op',
        'h:elp'
    ])

    def __init__(self, buche, reader, next_breakpoint=True):
        buche.configure('template', src=_template_path)
        self.dbf = buche['frames']
        self.db = buche['interact']
        self.db.html('<h3>Debug session start</h3>')
        self.db.markdown('Type `help` for a list of available commands.',
                         inline=True)
        self.next_breakpoint = next_breakpoint
        self.reader = reader

    async def command_step(self, vm, *args):
        """
        Step into the next instruction.
        """
        self.next_breakpoint = True
        return True

    async def command_next(self, vm, *args):
        """
        Execute the next instruction and stop.
        """
        self.next_breakpoint = vm.frame
        return True

    async def command_continue(self, vm, *args):
        """
        Continue until the next breakpoint or to the end.
        """
        self.next_breakpoint = False
        return True

    async def command_up(self, vm, *args):
        "TODO"
        self.db.log('TODO: UP FRAME', kind='error')

    async def command_down(self, vm, *args):
        "TODO"
        self.db.log('TODO: DOWN FRAME', kind='error')

    async def command_top(self, vm, n):
        """
        `top [n]`, inspect the n topmost elements of the stack.
        """
        if n.strip() == '':
            n = 1
        else:
            n = int(n)
        if n == 1:
            self.db(vm.frame.top())
        else:
            self.db(vm.frame.stack[-n:])

    async def command_var(self, vm, expr):
        """
        `v <var1> <var2> ...`, print out the value of each named
        variable. All versions of the variable will be shown.
        """
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
                self.db.markdown(f'Could not find a variable named `{vname}`',
                                 kind='error', inline=True)
            all_results.update(results)
        self.db.show(all_results, kind='result')

    async def command_help(self, vm, cmd):
        """
        List available commands.
        """
        class _:
            def __hrepr__(_, H, hrepr):
                t = H.table()
                t = t(H.tr(H.th("Command"), H.th("Description")))
                rev = {}
                for cmd, (canon, spec) in self.__commands__.items():
                    rev[canon] = spec
                for canon, spec in rev.items():
                    m = getattr(self, f'command_{canon}')
                    doc = m.__doc__ or "No documentation."
                    t = t(H.tr(H.td(canon), H.td(doc)))
                return t
        self.db(_())

    async def wait_for_command(self, vm):
        message = await self.reader.read_async()
        ctype = message.command
        if ctype == 'input':
            cmd, *rest = re.findall(r'\w+|\W+',
                                    message.contents.strip() or ' ')
            cmd = cmd.strip()
            arg = ''.join(rest)
            canon, _ = self.__commands__.get(cmd, None)
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

    async def error(self, vm, error):
        self.db.log('An error occurred!', kind='error')
        self.db.show(error, kind='error')
        self.dbf(VMPrinter(vm))
        await self.wait_for_command(vm)

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
