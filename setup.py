import os
from fnmatch import fnmatchcase
from distutils.util import convert_path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def find_packages(where='.', exclude=()):
    out = []
    stack = [(convert_path(where), '')]
    while stack:
        where, prefix = stack.pop(0)
        for name in os.listdir(where):
            fn = os.path.join(where, name)
            if ('.' not in name and os.path.isdir(fn) and
                    os.path.isfile(os.path.join(fn, '__init__.py'))):
                out.append(prefix + name)
                stack.append((fn, prefix + name + '.'))
    for pat in list(exclude) + ['ez_setup', 'distribute_setup']:
        out = [item for item in out if not fnmatchcase(item, pat)]
    return out


setup(name='Myia',
      version='0.0.1dev0',
      packages=find_packages(),
      install_requires=['buche', 'hrepr', 'numpy', 'unification', 'networkx', 'mypy', 'pep8'],
      )
