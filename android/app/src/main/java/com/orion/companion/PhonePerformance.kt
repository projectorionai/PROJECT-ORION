package com.orion.companion

import android.content.Context
import android.os.SystemClock
import java.util.Locale

/** Private timing diagnostics; never stores prompts, transcripts, keys or audio. */
internal object PhonePerformance {
    private const val STORE = "orion_phone_performance"
    private const val MAX_AGE_MS = 24 * 60 * 60 * 1000L

    fun now(): Long = SystemClock.elapsedRealtime()

    fun record(context: Context, name: String, startedAt: Long) {
        if (name !in setOf("capture", "transcription", "preparation", "first_text", "reply", "first_audio")) return
        val elapsed = (now() - startedAt).coerceIn(0L, 600_000L)
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putLong(name, elapsed).putLong("${name}_at", System.currentTimeMillis()).apply()
    }

    fun route(context: Context, route: String) {
        if (route !in setOf("Background", "Microphone", "Typed")) return
        val edit = context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
        for (name in listOf("capture", "transcription", "preparation", "first_text", "reply", "first_audio", "model", "voice_route", "voice_model", "context_chars", "speech_delivery", "voice_fallback")) {
            edit.remove(name).remove("${name}_at")
        }
        edit.remove("attempts").putString("route", route)
            .putLong("route_at", System.currentTimeMillis()).apply()
    }

    fun model(context: Context, model: String, attempts: Int) {
        if (!Regex("[A-Za-z0-9._-]{1,100}").matches(model)) return
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putString("model", model).putInt("attempts", attempts.coerceIn(0, 20))
            .putLong("model_at", System.currentTimeMillis()).apply()
    }

    fun context(context: Context, characters: Int) {
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putInt("context_chars", characters.coerceIn(0, PhonePromptWindow.MAX_CHARS))
            .putLong("context_chars_at", System.currentTimeMillis()).apply()
    }

    fun voice(context: Context, route: String, model: String) {
        if (route !in setOf("Live", "Direct TTS", "Phone", "Desktop TTS") ||
            !Regex("[A-Za-z0-9._-]{1,100}").matches(model)) return
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putString("voice_route", route).putString("voice_model", model)
            .putLong("voice_route_at", System.currentTimeMillis()).apply()
    }

    fun speechDelivery(context: Context, requests: Int, removedSamples: Long) {
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putInt("speech_requests", requests.coerceIn(0, 10_000))
            .putLong("speech_trim_ms", (removedSamples * 1000L / 24_000L).coerceAtLeast(0L))
            .putLong("speech_delivery_at", System.currentTimeMillis()).apply()
    }

    fun voiceFallback(context: Context, reason: String) {
        if (reason !in setOf("Selected", "Unavailable", "Partial audio")) return
        context.getSharedPreferences(STORE, Context.MODE_PRIVATE).edit()
            .putString("voice_fallback", reason).putLong("voice_fallback_at", System.currentTimeMillis()).apply()
    }

    fun summary(context: Context): String {
        val prefs = context.getSharedPreferences(STORE, Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        fun recent(name: String): Boolean = now - prefs.getLong("${name}_at", 0L) in 0..MAX_AGE_MS
        fun duration(name: String, label: String): String? =
            if (recent(name)) "$label: ${"%.2f".format(Locale.UK, prefs.getLong(name, 0L) / 1000.0)} s"
            else null
        val rows = mutableListOf<String>()
        if (recent("route")) rows += "Last route: ${prefs.getString("route", "Unknown")}"
        duration("capture", "Speech capture")?.let(rows::add)
        duration("transcription", "Transcription")?.let(rows::add)
        duration("preparation", "AI preparation")?.let(rows::add)
        duration("first_text", "Request to first text")?.let(rows::add)
        duration("reply", "Complete AI reply")?.let(rows::add)
        duration("first_audio", "Voice to first sound")?.let(rows::add)
        if (recent("voice_route")) rows += "Voice: ${prefs.getString("voice_route", "Unknown")} " +
            "(${prefs.getString("voice_model", "Unknown")})"
        if (recent("speech_delivery")) rows += "Speech requests: ${prefs.getInt("speech_requests", 0)}; " +
            "silence shortened: ${prefs.getLong("speech_trim_ms", 0)} ms"
        if (recent("voice_fallback")) rows += "Phone voice: ${prefs.getString("voice_fallback", "Unavailable")}"
        if (recent("model")) rows += "Text model: ${prefs.getString("model", "Unknown")} " +
            "(${prefs.getInt("attempts", 0)} attempt${if (prefs.getInt("attempts", 0) == 1) "" else "s"})"
        if (recent("context_chars")) rows += "AI context: ${prefs.getInt("context_chars", 0)} characters"
        return if (rows.isEmpty()) "No recent timing yet. Make a phone request, then reopen this panel."
        else rows.joinToString("\n") + "\n\nTimes are measured on this phone and contain no request text."
    }
}
