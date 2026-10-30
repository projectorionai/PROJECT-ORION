package com.orion.companion

internal data class PhoneOptionGroup(
    val title: String,
    val hint: String,
    val marker: String,
    val actions: IntArray,
)

/** Keeps every phone control reachable as the presentation changes. */
internal object PhoneOptionsMenu {
    val groups = listOf(
        PhoneOptionGroup("Phone", "Apps, maps and screen", "P", intArrayOf(0, 3, 4, 5, 15, 21, 23)),
        PhoneOptionGroup("Voice", "Listening and speech", "V", intArrayOf(1, 2, 11, 24, 16, 22)),
        PhoneOptionGroup("Connections", "AI, MCP and PC", "C", intArrayOf(6, 7, 8, 9, 10, 13, 17, 20, 25)),
        PhoneOptionGroup("Privacy", "History and diagnostics", "S", intArrayOf(12, 14, 18, 19)),
    )
}
