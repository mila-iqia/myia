
import os
import sys

from scripts.regen import regen_operations

here = os.path.dirname(os.path.realpath(__file__))


def parse_subs(subs):
    return dict(sub.split('=') for sub in subs)


def template(type, subs):
    contents = open(f'{here}/{type}.py.template').read()
    for key, repl in subs.items():
        contents = contents.replace(key, repl)
    return contents


def make_op(subs):
    subs = parse_subs(subs)
    name = subs['OPERATION']
    op_contents = template('op', subs)
    test_contents = template('test_op', subs)

    filename = f'./myia/operations/op_{name}.py'
    open(filename, 'w').write(op_contents)
    print('Wrote', filename)

    filename = f'./tests/operations/test_op_{name}.py'
    open(filename, 'w').write(test_contents)
    print('Wrote', filename)

    regen_operations()


def make_macro(subs):
    subs = parse_subs(subs)
    name = subs['OPERATION']
    macro_contents = template('macro', subs)
    test_contents = template('test_op', subs)

    filename = f'./myia/operations/macro_{name}.py'
    open(filename, 'w').write(macro_contents)
    print('Wrote', filename)

    filename = f'./tests/operations/test_macro_{name}.py'
    open(filename, 'w').write(test_contents)
    print('Wrote', filename)

    regen_operations()


def make_prim(subs):
    subs = parse_subs(subs)
    name = subs['OPERATION']
    prim_contents = template('prim', subs)
    test_contents = template('test_op', subs)

    filename = f'./myia/operations/prim_{name}.py'
    open(filename, 'w').write(prim_contents)
    print('Wrote', filename)

    filename = f'./tests/operations/test_prim_{name}.py'
    open(filename, 'w').write(test_contents)
    print('Wrote', filename)

    # We have to append these placeholders, because regen_operations will not
    # work otherwise.
    filename = f'./myia/operations/primitives.py'
    line = f"{name} = PlaceholderPrimitive('{name}')\n"
    open(filename, 'a').write(line)

    filename = f'./myia/operations/__init__.py'
    line = f"{name} = Operation(name='{name}')\n"
    open(filename, 'a').write(line)

    regen_operations()


if __name__ == '__main__':
    type, *subs = sys.argv[1:]
    globals()[f'make_{type}'](subs)
