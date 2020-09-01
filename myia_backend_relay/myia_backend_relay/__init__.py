__version__ = "0.1.0"

from myia.testing.multitest import register_backend_testing

from .relay import load_backend, load_options

register_backend_testing("relay", "cpu", {"target": "cpu", "device_id": 0})
register_backend_testing(
    "relay", "gpu", {"target": "cuda", "device_id": 0}, "relay-cuda"
)
