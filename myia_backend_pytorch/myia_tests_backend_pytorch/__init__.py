"""Configuration module for myia pytorch backend testing."""
from myia.testing.multitest import register_backend_testing

register_backend_testing("pytorch", "cpu", {"device": "cpu"})
register_backend_testing("pytorch", "gpu", {"device": "cuda"}, "pytorch-cuda")
