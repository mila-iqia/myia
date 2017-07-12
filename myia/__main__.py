"""Myia

Usage:
  myia parse [FILE] [-e <expr>]
  myia eval [FILE] [-e <expr>] [--args <args>]
  myia (-h | --help)
  myia --version

Options:
  -h --help     Show this screen.
  -e <expr>     Provide an expression to evaluate.
  --args <args> Provide arguments to the function to evaluate.
  --version     Show version.
"""

import sys
from docopt import docopt
from .front import parse_source
from .interpret import evaluate


def shame():
    raise NotImplementedError('You provided a command to myia that is not yet implemented. This is clearly the fault of myia developers. They are terrible. You should tell them.')

def getcode(arguments):
    expr, file = arguments['-e'], arguments['FILE']
    if expr and file:
        raise ValueError('Cannot provide a FILE and an expression at the same time.')
    code = expr if expr else open(file).read()
    url = '<command-line>' if expr else file
    return url, code

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Myia 0.0.0')
    # print(arguments)
    if arguments['parse']:
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
    elif arguments['eval']:
        url, code = getcode(arguments)
        r, bindings = parse_source(url, 1, code)
        result = evaluate(r, bindings)
        if arguments['--args']:
            print(result(*eval(arguments['--args'])))
        else:
            print(result)
    else:
        shame()
