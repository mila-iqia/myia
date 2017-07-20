
import argparse
import sys
from .compile import a_normal
from .front import parse_source
from .interpret import evaluate


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
p_parse.add_argument('--format', '-f', default='ascii',
                     help='Format to print out (ascii (default) or html)')

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


def shame():
    raise NotImplementedError(
        'You provided a command to myia that is not yet implemented.'
    )


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


def command_None(arguments):
    parser.print_help()


def command_parse(arguments):
    wrap = a_normal if arguments.a else (lambda x: x)
    url, code = getcode(arguments)
    r, bindings = parse_source(url, 1, code)

    ishtml = arguments.format == 'html'

    if bindings:
        if ishtml:
            print(H({k: wrap(v) for k, v in bindings.items()}).as_page())
        else:
            for k, v in bindings.items():
                print(f'{k}:\n  {wrap(v)}')
    else:
        if not isinstance(r, list):
            r = [r]
        for entry in r:
            if ishtml:
                print(H(wrap(entry)).as_page())
            else:
                print(wrap(entry))


def command_eval(arguments):
    url, code = getcode(arguments)
    r, bindings = parse_source(url, 1, code)
    result = evaluate(r, bindings)
    if arguments.args:
        args = eval(arguments.args)
        if not isinstance(args, tuple):
            args = args,
        print(result(*args))
    else:
        print(result)


if __name__ == '__main__':
    args = parser.parse_args()
    command = args.command
    method_name = 'command_{}'.format(command)
    if method_name not in globals():
        shame()
    else:
        globals()[method_name](args)
