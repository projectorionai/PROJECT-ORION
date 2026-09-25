"""Check installed distribution metadata without importing or installing packages."""
from importlib import metadata
from packaging.requirements import Requirement
from orion_core.data import ToolResult


def get_tool_schema():
    return {
        "name": "dependency_resolver",
        "description": "Checks installed package versions against supplied requirements, or an installed module distribution. Does not install packages.",
        "parameters": {
            "type": "object",
            "properties": {
                "module": {"type": "string"},
                "dependencies": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["module"]
        }
    }

def run(module, dependencies=None):
    try:
        if not isinstance(module, str) or not module.strip():
            raise ValueError("module must be non-empty text")
        if dependencies is None:
            dependencies = metadata.requires(module) or []
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise ValueError("dependencies must be a list of requirement strings")
        results = []
        for text in dependencies:
            req = Requirement(text)
            if req.url:
                raise ValueError("URL requirements cannot be verified from installed version metadata")
            if req.marker and not req.marker.evaluate():
                results.append({"package": req.name, "status": "not applicable"})
                continue
            try:
                version = metadata.version(req.name)
                status = "satisfied" if not req.specifier or req.specifier.contains(version, prereleases=True) else "version mismatch"
            except metadata.PackageNotFoundError:
                version, status = None, "missing"
            results.append({"package": req.name, "installed_version": version, "status": status})
        ok = all(row["status"] in {"satisfied", "not applicable"} for row in results)
        summary = "; ".join(row["package"] + ": " + row["status"] for row in results) or "No requirements declared."
        return ToolResult(summary + " No packages were installed.", ok=ok, evidence=results)
    except Exception as exc:
        return ToolResult(f"Dependency verification failed ({type(exc).__name__}). No packages were installed.", ok=False)
