
import os
import sys

errmsg = """ERROR: The buche command is not installed.

The buche command is a language-agnostic application and has to be
installed separately from the Python package. You can find the latest
release for your OS here:

    https://github.com/breuleux/buche/releases/latest

Or if npm is installed, you can run the command:

    npm install buche -g
"""

if __name__ == '__main__':
    os.environ['PYTHONBREAKPOINT'] = 'buche.breakpoint'
    cmd = 'from debug.run import main; main()'
    try:
        os.execvp('buche', ['buche', 'python3', '-u', '-c', cmd]
                  + sys.argv[1:])
    except FileNotFoundError:
        print(errmsg, file=sys.stderr)
        sys.exit(1)
