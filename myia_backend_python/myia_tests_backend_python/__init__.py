"""Configuration module for myia Python backend testing."""
from myia.testing.multitest import register_backend_testing

register_backend_testing("python", "cpu", {})
