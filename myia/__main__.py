
import argparse
import sys
import traceback
from .compile import a_normal
from .front import parse_source
from .interpret import evaluate
from .validate import grad_test, unbound
from .buche import buche


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

p_grad = subparsers.add_parser(
    'grad',
    help='Display the gradient of the expression.'
)
p_grad.add_argument(
    'FILE',
    help='The file with the function to take the gradient of'
)
p_grad.add_argument('--expr', '-e', metavar='EXPR',
                    dest='expr',
                    help='The expression to take the gradient of.')
p_grad.add_argument('--format', '-f', default='text',
                    help='Format to print out (text (default) or html)')
p_grad.add_argument('--args', metavar='ARGS',
                    dest='args',
                    help='Arguments to provide to the function.')


def shame():
    raise NotImplementedError(
        'You provided a command to myia that is not yet implemented.'
    )


setup_done = False


def display(data, format, mode='normal'):
    global setup_done
    if format == 'html':
        print(H(data).as_page())
    elif format == 'buche':
        if not setup_done:
            buche.raw(command='open', path='/', type='tabs', anchor='top')
            buche.open('funcs', 'tabs', anchor='left')
            setup_done = True
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
    code = expr if expr else open(file).read()
    url = '<command-line>' if expr else file
    return url, code


def getargs(arguments):
    if arguments.args:
        args = eval(arguments.args)
        if not isinstance(args, tuple):
            args = args,
        return args
    else:
        return None


def command_None(arguments):
    parser.print_help()


def command_parse(arguments):
    def wrap(x):
        unbound(x)
        if arguments.a:
            return a_normal(x)
        else:
            return x
    url, code = getcode(arguments)
    r, bindings = parse_source(url, 1, code)
    if bindings:
        display({k: wrap(v) for k, v in bindings.items()},
                arguments.format)
    else:
        if not isinstance(r, list):
            r = [r]
        for entry in r:
            display(wrap(entry), arguments.format)


def command_eval(arguments):
    url, code = getcode(arguments)
    r, bindings = parse_source(url, 1, code)
    result = evaluate(r, bindings)
    args = getargs(arguments)
    if args:
        try:
            value = result(*args)
            display(result(*args), arguments.format)
        except Exception as exc:
            traceback.print_exc()
            display({k: v for k, v in bindings.items()},
                    arguments.format)
    else:
        display(result, arguments.format)


def command_grad(arguments):
    url, code = getcode(arguments)
    results = grad_test((url, 1, code))
    args = getargs(arguments)
    bindings = {**results['func_bindings'], **results['grad_bindings']}
    if args:
        t = results['test']
        try:
            rval = t(args)
            display(rval, arguments.format)
        except Exception as exc:
            rval = None
            traceback.print_exc()
        display(bindings, arguments.format, 'bindings')
    else:
        for k, v in bindings.items():
            unbound(v)
        display(bindings, arguments.format, 'bindings')


if __name__ == '__main__':
    args = parser.parse_args()
    command = args.command
    method_name = 'command_{}'.format(command)
    if method_name not in globals():
        shame()
    else:
        globals()[method_name](args)
