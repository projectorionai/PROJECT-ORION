package com.orion.companion

internal data class PhonePromptLine(val role: String, val text: String)

/** Keep recent context useful without resending an entire encrypted chat log. */
internal object PhonePromptWindow {
    const val MAX_CHARS = 32_000
    private const val MAX_MESSAGES = 16
    private const val MAX_OLD_ENTRY = 8_000

    fun select(history: List<PhonePromptLine>): List<PhonePromptLine> {
        val selected = mutableListOf<PhonePromptLine>()
        var used = 0
        for (index in history.indices.reversed()) {
            if (selected.size >= MAX_MESSAGES || used >= MAX_CHARS) break
            val row = history[index]
            val remaining = MAX_CHARS - used
            val limit = if (index == history.lastIndex && row.role == "user") remaining
                else minOf(remaining, MAX_OLD_ENTRY)
            if (limit < 64) break
            val excerpt = when {
                row.text.length <= limit -> row.text
                row.role == "model" -> {
                    val note = "[Earlier reply omitted]\n"
                    note + row.text.takeLast(limit - note.length)
                }
                else -> {
                    val note = "[Earlier request omitted]\n"
                    note + row.text.take(limit - note.length)
                }
            }
            selected += PhonePromptLine(row.role, excerpt)
            used += excerpt.length
        }
        return selected.asReversed()
    }
}
