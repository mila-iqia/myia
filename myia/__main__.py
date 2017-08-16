"""
Command-line interface to Myia, mostly for development and
testing at the moment. To get help, run the following command:

$ python -m myia -h
"""

import argparse
import sys
import traceback
from importlib import import_module
from . import ast
from .compile import a_normal
from .front import parse_source
from .interpret import evaluate
from .validate import \
    unbound, missing_source, \
    analysis
from .buche import buche, Reader, id_registry

from .event import on_discovery
from .interpret import VM
from .ast import Symbol
from .front import ParseEnv
from .debug import BucheDb


###############################
# Argument parser definitions #
###############################

parser = argparse.ArgumentParser(prog='myia')
subparsers = parser.add_subparsers(dest='command')

p_parse = subparsers.add_parser(
    'parse',
    help='Parse an expression and print its representation'
)
p_parse.add_argument('FILE', nargs='?', help='The file to parse.')
p_parse.add_argument('--expr', '-e', metavar='EXPR',
                     dest='expr', help='The expression to parse.')
p_parse.add_argument('-a', action='store_true',
                     help='Convert to a-normal form.')
p_parse.add_argument('--format', '-f', default='text',
                     help='Format to print out (text (default) or html)')

p_eval = subparsers.add_parser('eval', help='Evaluate an expression')
p_eval.add_argument('FILE', nargs='?', help='The file to evaluate.')
p_eval.add_argument(
    '--expr',
    '-e',
    metavar='EXPR',
    dest='expr',
    help='The expression to evaluate.'
)
p_eval.add_argument('--args', metavar='ARGS',
                    dest='args',
                    help='Arguments to provide to the function.')
p_eval.add_argument('--format', '-f', default='text',
                    help='Format to print out (text (default) or html)')

# p_grad = subparsers.add_parser(
#     'grad',
#     help='Display the gradient of the expression.'
# )
# p_grad.add_argument(
#     'FILE',
#     help='The file with the function to take the gradient of'
# )
# p_grad.add_argument('--expr', '-e', metavar='EXPR',
#                     dest='expr',
#                     help='The expression to take the gradient of.')
# p_grad.add_argument('--format', '-f', default='text',
#                     help='Format to print out (text (default) or html)')
# p_grad.add_argument('--args', metavar='ARGS',
#                     dest='args',
#                     help='Arguments to provide to the function.')

p_inspect = subparsers.add_parser('inspect',
                                  help='Inspect/evaluate an expression')
p_inspect.add_argument('FILE', nargs='?', help='The file to evaluate.')
p_inspect.add_argument(
    '--expr',
    '-e',
    metavar='EXPR',
    dest='expr',
    help='The expression to evaluate.'
)
p_inspect.add_argument('--args', metavar='ARGS', default='()',
                       dest='args',
                       help='Arguments to provide to the function.')
p_inspect.add_argument('--mode', dest='mode', default='eval',
                       help='One of: eval')
p_inspect.add_argument('--stores', action='store_true',
                       help='Log the values taken by each variable.')
p_inspect.add_argument('--decls', action='store_true',
                       dest='decls',
                       help='Log the AST for all functions.')
p_inspect.add_argument('--all', action='store_true',
                       help='Log everything that can be logged.')
p_inspect.add_argument('--check', metavar='PROBLEMS', dest='check',
                       help='One or more of: unbound,source.')


p_debug = subparsers.add_parser('debug',
                                help='debug/evaluate an expression')
p_debug.add_argument('FILE', nargs='?', help='The file to evaluate.')
p_debug.add_argument(
    '--expr',
    '-e',
    metavar='EXPR',
    dest='expr',
    help='The expression to evaluate.'
)
p_debug.add_argument('--args', metavar='ARGS', default='()',
                     dest='args',
                     help='Arguments to provide to the function.')


####################
# Helper functions #
####################

burepl = buche.open('repl', 'log', hasInput=True, force=True)
reader = Reader()
budb = BucheDb(burepl, reader)


def setup_buche(arguments):
    """
    Set up buche tabs for the inspect command.
    """
    # buche.raw(command='open', path='/', type='tabs', anchor='top')

    # This is where print(...) statements will end up.
    buche.open('_', 'log', force=True)

    # decls will list all Myia functions, including
    # auxiliary ones, as they are compiled. --decls option.
    buche.open('decls', 'tabs', force=True, anchor='left')

    # stores will log the values of each variable through
    # execution (not cheap) --stores option.
    buche.open('stores', 'tabs', force=True, anchor='left')

    # problems will list certain problems in the code,
    # e.g. unbound variables or nodes that lack a source,
    # and more in the future. --checks option.
    buche.open('problems', 'tabs', force=True, anchor='left')

    def add_class(node, kls):
        # Helper function to retroactively add a CSS class
        # to all the printouts of the given node. These
        # already have the pyid-{id(node)} class, so we
        # can select them that way.
        buche.raw(command = 'reprocess',
                  selector = f'.pyid-{id(node)}',
                  body = f'this.classList.add("{kls}")')

    # Set up on_discovery hooks to log everything we need.

    if arguments.stores:
        @on_discovery(VM)
        def on_instruction_store(_, frame, node, var):
            buche['stores'][str(var).replace('/', '.')](frame.top())

    if arguments.check:
        checks = set(arguments.check.split(','))

        @on_discovery(ParseEnv)
        def on_declare(e, name, value):
            pbuche = buche['problems']
            url = e.owner.url.split('/')[-1]
            if 'unbound' in checks:
                for node in unbound(value):
                    node.annotations = node.annotations | {'unbound'}
                    pbuche['unbound'](node)
            if 'source' in checks:
                for node in missing_source(value):
                    node.annotations = node.annotations | {'missing_source'}
                    msbuche = pbuche['missing_source']
                    msbuche(node)
                    if node.trace:
                        t = node.trace[-1]
                        msbuche.pre('  Definition at:')
                        msbuche.pre(f'    {t.filename} line {t.lineno}')
                        msbuche.pre(f'    {t.line}')

    if arguments.decls:
        @on_discovery(ParseEnv)
        def on_declare(e, name, value):
            buche['decls'][str(name)](value)

        @on_discovery(VM)
        def on_error(e, exc):
            vm = e.owner
            focus = vm.frame.focus
            add_class(focus, 'error0')
            buche.html('<h2>An error occurred</h2>')
            buche.html('<h3>Node</h3>')
            buche(vm.frame)
            buche(focus)
            buche.html('<h3>Traceback</h3>')
            for i, frame in enumerate([vm.frame] + vm.frames):
                node = frame.focus
                if node:
                    add_class(node, 'error')


def H(node):
    try:
        from hrepr import hrepr
    except ImportError:
        print('The --format html option requires the \'hrepr\' package',
              'to be installed.\nTo install the package, use the command:',
              '\n\n$ pip3 install hrepr',
              file=sys.stderr)
        sys.exit(1)
    return hrepr(node)


def shame():
    raise NotImplementedError(
        'You provided a command to myia that is not yet implemented.'
    )


def display(data, format, mode='normal'):
    global setup_done
    if format == 'html':
        print(H(data).as_page())
    elif format == 'buche':
        if mode == 'normal':
            buche(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                buche['funcs'][str(k)](v)
        else:
            buche(data)
    elif isinstance(data, dict):
        for k, v in data.items():
            print(f'{k}:\n  {v}')
    else:
        print(data)


def getcode(arguments):
    expr, file = arguments.expr, arguments.FILE
    if expr and file:
        raise ValueError(
            'Cannot provide a FILE and an expression at the same time.'
        )
    if not expr and not file:
        print('Must provide FILE or use -e option.', file=sys.stderr)
        sys.exit(1)
    if expr:
        return '<command-line>', 1, expr
    elif ':' in file:
        file, name = file.split(':')
        m = import_module(file)
        f = getattr(m, name)
        return getattr(f, '__orig__', f)
    else:
        code = open(file).read()
        url = file
        return url, 1, code


def getfn(arguments):
    data = getcode(arguments)
    if isinstance(data, tuple):
        return parse_source(*data)
    else:
        return parse_function(data)


def getargs(arguments):
    if arguments.args:
        args = eval(arguments.args)
        if not isinstance(args, tuple):
            args = args,
        return args
    else:
        return None

############
# COMMANDS #
############

# $ python -m myia <command> <arguments>
# ==> executes command_<command>(<arguments>)


def command_None(arguments):
    parser.print_help()


def command_parse(arguments):
    def wrap(x):
        if arguments.a:
            return a_normal(x)
        else:
            return x
    r, genv = getfn(arguments)
    bindings = genv.bindings
    if bindings:
        display({k: wrap(v) for k, v in bindings.items()},
                arguments.format)
    else:
        if not isinstance(r, list):
            r = [r]
        for entry in r:
            display(wrap(entry), arguments.format)


def command_eval(arguments):
    r, genv = getfn(arguments)
    result = evaluate(r, genv)
    args = getargs(arguments)
    if args:
        value = result(*args)
        display(value, arguments.format)
    else:
        display(result, arguments.format)


# def command_grad(arguments):
#     url, code = getcode(arguments)
#     results = grad_test((url, 1, code))
#     args = getargs(arguments)
#     bindings = {**results['func_bindings'], **results['grad_bindings']}
#     if args:
#         t = results['test']
#         try:
#             rval = t(args)
#             display(rval, arguments.format)
#         except Exception as exc:
#             rval = None
#             traceback.print_exc()
#         display(bindings, arguments.format, 'bindings')
#     else:
#         for k, v in bindings.items():
#             unbound(v)
#         display(bindings, arguments.format, 'bindings')


def command_inspect(arguments):
    ast.__save_trace__ = True
    if arguments.all:
        arguments.decls = True
        arguments.stores = True
    setup_buche(arguments)
    code = getcode(arguments)
    args = getargs(arguments)
    if args:
        value = analysis(arguments.mode, code, args)
        buche['_'].html(f'<h2>Results for: {arguments.mode}</h2>')
        buche['_'](value)
    else:
        buche['_']('Done')

    reader = Reader()

    @reader.on_click
    def handle(e, cmd):
        try:
            obj = id_registry[int(cmd.objId)]
            # buche[cmd.path](obj.about)
            # buche[cmd.path](obj.trace)
            # buche[cmd.path](obj.find_location())
            from .ast import AboutPrinter
            buche[cmd.path](AboutPrinter(obj))
        except Exception as exc:
            buche[cmd.path](exc)
    reader.run()


def command_debug(arguments):
    ast.__save_trace__ = True
    arguments.decls = True
    arguments.stores = False
    arguments.check = False
    setup_buche(arguments)
    args = getargs(arguments)
    r, genv = getfn(arguments)

    @reader.on_click
    def handle(e, cmd):
        try:
            obj = id_registry[int(cmd.objId)]
            if 'break' in obj.annotations:
                obj.annotations.remove('break')
                buche[cmd.path]('Unset breakpoint.')
            else:
                obj.annotations.add('break')
                buche[cmd.path]('Set breakpoint.')
        except Exception as exc:
            buche[cmd.path](exc)

    budb.set_trace()
    fn = evaluate(genv[r])
    try:
        buche(fn.debug(args, budb), kind='result')
    except Exception as exc:
        buche(exc, kind='error')


if __name__ == '__main__':
    args = parser.parse_args()
    command = args.command
    method_name = 'command_{}'.format(command)
    if method_name not in globals():
        shame()
    else:
        globals()[method_name](args)
