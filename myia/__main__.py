"""Myia

Usage:
  myia parse [FILE] [-e <expr>]
  myia (-h | --help)
  myia --version

Options:
  -h --help     Show this screen.
  -e <expr>     Provide an expression to evaluate.
  --version     Show version.
"""

import sys
from docopt import docopt
from .front import parse_source


def shame():
    raise NotImplementedError('You provided a command to myia that is not yet implemented. This is clearly the fault of myia developers. They are terrible. You should tell them.')


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Myia 0.0.0')
    # print(arguments)
    if arguments['parse']:
        expr, file = arguments['-e'], arguments['FILE']
        if expr and file:
            raise ValueError('Cannot provide a FILE and an expression at the same time.')
        text = expr if expr else open(file).read()
        url = '<command-line>' if expr else file
        print(parse_source(url, 1, text))
    else:
        shame()
