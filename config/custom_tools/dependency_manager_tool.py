"""Inspect requirements and install them only when explicitly requested."""
import subprocess
import sys
from pathlib import Path
from orion_core.data import ToolResult


def get_tool_schema():
    return {
        "name": "dependency_manager",
        "description": "Automates the identification and installation of dependencies for module forging.",
        "parameters": {
            "type": "object",
            "properties": {
                "module_name": {"type": "string", "description": "The name of the module to check dependencies for."},
                "install": {"type": "boolean", "description": "Whether to install the dependencies if they are missing."}
            },
            "required": ["module_name"]
        }
    }


def run(module_name, install=False):
    try:
        if type(install) is not bool:
            return ToolResult("install must be a boolean.", ok=False)
        requirements_file = (Path(module_name) / "requirements.txt").resolve()
        if not requirements_file.is_file():
            return ToolResult("No requirements.txt found for the supplied module.", ok=False)
        dependencies = [line.strip() for line in requirements_file.read_text(encoding="utf-8-sig").splitlines()
                        if line.strip() and not line.lstrip().startswith("#")]
        if not dependencies:
            return ToolResult("The requirements file has no dependencies.")
        if not install:
            return ToolResult("Declared requirements (installation not checked): " + ", ".join(dependencies))
        if getattr(sys, "frozen", False):
            return ToolResult("Dependency installation requires the source Python environment; the standalone app cannot run pip.", ok=False)
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                                 "-r", str(requirements_file)], cwd=str(requirements_file.parent),
                                capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            # pip output can echo credentials embedded in private package URLs.
            return ToolResult(f"Dependency installation failed (pip exit {result.returncode}); dependencies may be partially installed.", ok=False)
        return ToolResult("pip completed successfully for the requirements file. Module execution has not been verified.",
                          evidence=[{"pip_exit_code": 0, "module_tested": False}])
    except subprocess.TimeoutExpired:
        return ToolResult("Dependency installation timed out; dependencies may be partially installed.", ok=False)
    except Exception as exc:
        return ToolResult(f"Dependency check failed ({type(exc).__name__}).", ok=False)
