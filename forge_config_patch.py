
# forge_config_patch.py
# Corrective configuration script to resolve 'No code generator configured' error.

import os
from dataclasses import dataclass, field

@dataclass
class ForgeConfig:
    """Configuration specifically for the Forge self-improvement engine."""
    code_generator: str = "Claude-3.5-Sonnet" # Explicitly setting the code generator
    status: str = "active"
    
    def __post_init__(self) -> None:
        print(f"FORGE CONFIGURATION PATCH APPLIED: Code generator set to {self.code_generator}")

# This patch must be dynamically loaded by the main application context to enable the forge engine.
