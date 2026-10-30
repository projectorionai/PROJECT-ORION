"""Discover installed modules without importing their package initializers."""

from __future__ import annotations

import sys


def module_present(name: str) -> bool:
    """Inspect import specs, including dotted names and frozen importers.

    ``importlib.util.find_spec('package.child')`` imports ``package`` first.
    Diagnostics must not run optional native initializers on their worker.
    Walk the normal meta-path finders with each package's search locations
    instead. A loaded package's actual path takes precedence over its spec.
    This is dependency evidence only: no loader or module body is executed.
    """
    if not name or any(not part for part in name.split(".")):
        return False
    if name in sys.modules:
        return sys.modules[name] is not None
    parts = name.split(".")
    search_path = None
    try:
        for index in range(len(parts)):
            fullname = ".".join(parts[:index + 1])
            if fullname in sys.modules:
                loaded = sys.modules[fullname]
                if loaded is None:
                    return False
                if index == len(parts) - 1:
                    return True
                search_path = vars(loaded).get("__path__")
            else:
                spec = None
                for finder in tuple(sys.meta_path):
                    spec = finder.find_spec(fullname, search_path)
                    if spec is not None:
                        break
                if spec is None:
                    return False
                if index == len(parts) - 1:
                    return True
                search_path = spec.submodule_search_locations
            if search_path is None:
                return False
    except Exception:
        # Import-hook failure is not evidence of an installed dependency.
        return False
    return False
