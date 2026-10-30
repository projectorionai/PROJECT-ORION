package com.orion.companion

/** Explicit, user-tapped actions handled by the foreground phone Activity. */
object SentinelActionContract {
    const val EXTRA_ACTION = "com.orion.companion.extra.SENTINEL_ACTION"
    const val DICTATE = "dictate"
    const val TOGGLE_LISTENING = "toggle_listening"
    const val SCREEN_SHARING = "screen_sharing"
    const val PHONE_CONTROLS = "phone_controls"
    const val OPEN_ORION = "open_orion"

    private val actions = setOf(
        DICTATE, TOGGLE_LISTENING, SCREEN_SHARING, PHONE_CONTROLS, OPEN_ORION,
    )

    fun isSupported(action: String?) = action in actions
}

/** Pixel coordinates for overlays; all placements stay inside the usable display. */
internal object SentinelLayout {
    data class Bounds(val left: Int, val top: Int, val right: Int, val bottom: Int) {
        val width get() = (right - left).coerceAtLeast(0)
        val height get() = (bottom - top).coerceAtLeast(0)
    }

    data class Position(val x: Int, val y: Int)

    fun clamp(x: Int, y: Int, width: Int, height: Int, bounds: Bounds) = Position(
        x.coerceIn(bounds.left, maxOf(bounds.left, bounds.right - width)),
        y.coerceIn(bounds.top, maxOf(bounds.top, bounds.bottom - height)),
    )

    fun menu(
        orbX: Int, orbY: Int, orbWidth: Int, menuWidth: Int, menuHeight: Int,
        gap: Int, bounds: Bounds,
    ): Position {
        val right = orbX + orbWidth + gap
        val x = if (right + menuWidth <= bounds.right) right else orbX - menuWidth - gap
        return clamp(x, orbY, menuWidth, menuHeight, bounds)
    }
}
