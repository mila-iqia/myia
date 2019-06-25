"""Displays line locations in color."""

from colorama import Fore, Style

from ..utils import eprint


def _show_location(loc, label, mode='color', color='red'):
    with open(loc.filename, 'r') as contents:
        lines = contents.read().split('\n')
        _print_lines(lines, loc.line, loc.column,
                     loc.line_end, loc.column_end,
                     label, mode, color)


def _print_lines(lines, l1, c1, l2, c2, label='', mode='color', color='red'):
    for ln in range(l1, l2 + 1):
        line = lines[ln - 1]
        if ln == l1:
            trimmed = line.lstrip()
            to_trim = len(line) - len(trimmed)
            start = c1 - to_trim
        else:
            trimmed = line[to_trim:]
            start = 0

        if ln == l2:
            end = c2 - to_trim
        else:
            end = len(trimmed)

        if mode == 'color':
            prefix = trimmed[:start]
            hl = trimmed[start:end]
            rest = trimmed[end:]
            if color == 'red':
                eprint(f'{ln}: {prefix}{Fore.RED}{Style.BRIGHT}'
                       f'{hl}{Style.RESET_ALL}{rest}')
            elif color == 'magenta':
                eprint(f'{ln}: {prefix}{Fore.MAGENTA}{Style.BRIGHT}'
                       f'{hl}{Style.RESET_ALL}{rest}')
        else:
            eprint(f'{ln}: {trimmed}')
            prefix = ' ' * (start + 2 + len(str(ln)))
            eprint(prefix + '^' * (end - start) + label)
