"""Myia Relay backend.

   isort:skip_file
"""
__version__ = "0.1.0"

from .relay import load_backend, load_options
from myia.testing.multitest import register_backend_testing


register_backend_testing("relay", "cpu", {"target": "cpu", "device_id": 0})
register_backend_testing(
    "relay", "gpu", {"target": "cuda", "device_id": 0}, "relay-cuda"
)
