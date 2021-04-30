class PdbRunCall:
    """Helper class to run code with PDB.
    We want PDB to be able to display code using `list .` command, so
    we need to save code into a file and import it as a valid module later.
    """

    def __init__(self, code, name, symbols):
        """Initialize.
        :param code: code to compile
        :param name: name of function to run
        :param external_symbols: external symbols needed to execute the code.
            Dictionary mapping name to symbol
        """
        self.code = code
        self.name = name
        self.symbols = symbols or {}

    def __call__(self, *args):
        """Execute main function with given args."""
        import importlib
        import os
        import pdb
        import sys
        import tempfile

        # Create temporary code file.
        code_fd, code_path = tempfile.mkstemp(
            prefix="myia_backend_python_code_", suffix=".py"
        )
        # Get module directory and name.
        module_dir = os.path.dirname(code_path)
        module_name = os.path.splitext(os.path.basename(code_path))[0]
        # Add module to sys.path
        sys.path.append(module_dir)
        try:
            # Save code into module file.
            with open(code_path, "w") as code_file:
                code_file.write(self.code)
            # Import module.
            module = importlib.import_module(module_name)
            module.__dict__.update(self.symbols)
            # Run main function.
            output = pdb.runcall(getattr(module, self.name), *args)

        # NB: I don't know why, but code executed after PDB call is
        # systematically reported as uncovered by pytest-cov, so I am
        # excluding following lines from coverage.
        finally:  # pragma: no cover
            # Reset sys.path
            sys.path.remove(module_dir)
            # Close and delete code file.
            os.close(code_fd)
            os.remove(code_path)
        return output  # pragma: no cover