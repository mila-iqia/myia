
import argparse
import sys
from .front import parse_source
from .interpret import evaluate

parser = argparse.ArgumentParser(prog='myia')
subparsers = parser.add_subparsers(dest='command')

p_parse = subparsers.add_parser(
    'parse',
    help='Parse an expression and print its representation'
)
p_parse.add_argument('FILE', nargs='?', help='The file to parse.')
p_parse.add_argument('--expr', '-e', metavar='EXPR',
                     dest='expr', help='The expression to parse.')

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
    url, code = getcode(arguments)
    r, bindings = parse_source(url, 1, code)
    if bindings:
        for k, v in bindings.items():
            print('{}:\n  {}'.format(k, v))
    else:
        if not isinstance(r, list):
            r = [r]
        for entry in r:
            print(entry)


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
