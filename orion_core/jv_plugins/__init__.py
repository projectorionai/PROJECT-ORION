"""Reviewed ORION adapters for the user's JARVIS plugin collection.

Importing this package never opens devices, schedules work or contacts a service.
"""

from .runtime import configure, shutdown

__all__ = ["configure", "shutdown"]
