package com.orion.companion

/** Split long replies at natural pauses without consuming unnecessary speech requests. */
object SpeechText {
    /** Begin speech early only when the request cannot reasonably dispatch a phone action. */
    fun mayStreamAnswer(request: String): Boolean = !Regex(
        "\\b(?:open|launch|play|send|e-?mail|call|text|write|create|set|tap|click|scroll|" +
            "share|navigate|find|list|show|start|stop|run|read|move|delete|search|book|" +
            "schedule|message|remind|enable|disable|turn|unlock|lock|pause|resume|" +
            "install|uninstall|dial|take|record|capture|download|upload)\\b", RegexOption.IGNORE_CASE,
    ).containsMatchIn(request)

    /** Collect complete spoken phrases without repeating streamed text. */
    class Stream {
        private val pending = StringBuilder()
        private var started = false

        fun add(delta: String): List<String> {
            pending.append(delta)
            val complete = mutableListOf<String>()
            while (true) {
                // Start with one sentence, then group enough speech to cover
                // the next synthesis request. A paragraph is a visual boundary.
                val minimum = if (started) 219 else 23
                val sentence = (minimum until minOf(pending.length - 1, 600)).firstOrNull { index ->
                    pending[index] in ".!?" && pending[index + 1].isWhitespace()
                }?.plus(1)
                val cut = sentence ?: if (pending.length >= 600)
                    (580 downTo 400).firstOrNull { pending[it].isWhitespace() } else null
                if (cut == null) break
                spoken(pending.substring(0, cut)).takeIf { it.isNotEmpty() }?.let(complete::add)
                started = true
                pending.delete(0, cut)
                while (pending.isNotEmpty() && pending[0].isWhitespace()) pending.deleteCharAt(0)
            }
            return complete
        }

        fun finish(): String = spoken(pending.toString()).also { pending.clear() }
    }

    // The host rejects a speech request above 1,200 characters.
    fun parts(text: String, limit: Int = 1200): List<String> {
        require(limit >= 2)
        var remaining = spoken(text)
        val parts = mutableListOf<String>()
        while (remaining.length > limit) {
            var cut = -1
            for (index in limit - 1 downTo 0) {
                if (remaining[index] in ".!?\n" && remaining[index + 1].isWhitespace()) {
                    cut = index + 1
                    break
                }
            }
            if (cut < 0) cut = remaining.lastIndexOf(' ', limit).takeIf { it > 0 } ?: limit
            if (cut < remaining.length && remaining[cut - 1].isHighSurrogate() &&
                remaining[cut].isLowSurrogate()) cut -= 1
            parts.add(remaining.substring(0, cut).trim())
            remaining = remaining.substring(cut).trimStart()
        }
        if (remaining.isNotEmpty()) parts.add(remaining)
        return parts
    }

    private fun spoken(text: String): String = text.trim().replace(Regex("\\s+"), " ")

    /** Only these models accept the speech schema used by the phone. */
    fun voiceModels(configured: String): List<String> {
        val available = listOf(
            "gemini-3.8-flash-tts",
            "gemini-3.8-flash-lite-tts",
            "gemini-2.5-flash-preview-tts",
            "gemini-3.1-flash-tts-preview",
        )
        // The legacy preview is last because its small daily quota is often exhausted.
        return if (configured in available && configured != available.last())
            listOf(configured) + available.filterNot { it == configured }
        else available
    }
}
