"""Configuration module for myia relay backend testing."""
from myia.testing.multitest import register_backend_testing

register_backend_testing("relay", "cpu", {"target": "cpu", "device_id": 0})
register_backend_testing(
    "relay", "gpu", {"target": "cuda", "device_id": 0}, "relay-cuda"
)
