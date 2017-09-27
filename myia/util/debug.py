
import re
from ..stx.nodes import Symbol


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
            v = H.view(tab, pane)
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
        'vars;?*'
    ])

    def __init__(self, buche, reader):
        self.dbf = buche.open_slides('frame')
        self.db = buche.open_log('debug', force=True, hasInput=True)
        self.db.log('-- Debug --')
        self.reader = reader

    async def command_step(self, vm, policy, *args):
        return True

    async def command_next(self, vm, policy, *args):
        return vm.frame

    async def command_continue(self, vm, policy, *args):
        return False

    async def command_up(self, vm, policy, *args):
        self.db.log('TODO: UP FRAME', kind='error')

    async def command_down(self, vm, policy, *args):
        self.db.log('TODO: DOWN FRAME', kind='error')

    async def command_expression(self, vm, policy, expr):
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

    async def wait_for_command(self, vm, policy):
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
                # self.db.log(message.contents, kind='echo')
                self.db.log(f'Unknown command: {cmd}', kind='error')
            else:
                # self.db.log(f'{canon} {" ".join(args)}', kind='echo')
                method = getattr(self, f'command_{canon}')
                res = await method(vm, policy, arg)
                if res is not None:
                    return res
        return await self.wait_for_command(vm, policy)

    async def __call__(self, vm, policy):
        instr = vm.frame.current_instruction()
        # self.db(instr)
        if not instr or instr.command != 'reduce':
            return policy
        foc = vm.frame.focus
        if foc:
            # self.db(foc.find_location())
            # foc.annotations.add('error')
            # self.dbf(vm.frame.code.lbda or vm.frame.code.node)
            # foc.annotations.remove('error')
            self.dbf(VMPrinter(vm))
        else:
            self.db('No focus.')
        return await(self.wait_for_command(vm, policy))
