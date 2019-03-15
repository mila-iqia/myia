from ..pipeline import PipelineStep

from .backends import load_backend


class CompileStep(PipelineStep):
    """Step to compile a graph to a configurable backend.

    Inputs:
        graph: a graph (must be typed)

    Outputs:
        output: a callable

    """
    def __init__(self, pipeline_init, backend=None, backend_options=None):
        """Initialize a CompileStep.

        Arguments:
            backend: (str) the name of the backend to use
            backend_options: (dict) options for the backend

        """
        super().__init__(pipeline_init)
        self.backend = load_backend(backend)
        self.backend.init(**backend_options)
    
    def step(self, graph):
        """Compile the set of graphs."""
        out = self.backend.compile(graph)
        return {'output': out}


step_compile = CompileStep.partial()
