"""
GUI package — the Mark VIII dual-window Qt shell.

    style        — application-wide stylesheet
    widgets      — MetricBar, MiniOrb, HolographicToggle, ApiKeyDialog
    hud          — CentralHud (Liquid Vector Orb)
    views        — stacked views inside the core window
    core_window  — OrionCoreWindow: conversation, voice activity, system health
    dashboard    — WidgetDashboardWindow: productivity tools and agent controls
"""

from .command_centre import CommandCentreWindow            # noqa: F401
from .core_window import OrionCoreWindow, OrionMainWindow  # noqa: F401
from .dashboard import WidgetDashboardWindow                # noqa: F401
from .widgets import ApiKeyDialog, HolographicToggle        # noqa: F401
