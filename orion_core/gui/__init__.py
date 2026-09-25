"""
GUI package — the Mark VIII dual-window Qt shell.

    style        — application-wide stylesheet
    widgets      — MetricBar, MiniOrb, HolographicToggle, ApiKeyDialog
    hud          — CentralHud (Liquid Vector Orb)
    views        — LOG/MEMORY/TELEMETRY, now Command Deck pages (app.py)
    core_window  — OrionCoreWindow: face-only shell (everything else lives
                   on the Command Deck)
    dashboard    — WidgetDashboardWindow: productivity tools and agent controls
    ops_deck     — OperationsDeckView: Companion/Executive/Inner-Voice/
                   Environment/... panels on the OPS deck page
"""

from .command_centre import CommandCentreWindow            # noqa: F401
from .core_window import OrionCoreWindow, OrionMainWindow  # noqa: F401
from .dashboard import WidgetDashboardWindow                # noqa: F401
from .widgets import ApiKeyDialog, HolographicToggle        # noqa: F401
