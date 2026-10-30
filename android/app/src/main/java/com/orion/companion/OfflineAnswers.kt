package com.orion.companion

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale

/**
 * The few questions the phone can answer itself with no internet and no PC:
 * the time, the date and the battery. Everything else needs the hosted AI.
 * Only a short, plain question matches - "what time is the meeting" is not
 * "what time is it".
 */
internal object OfflineAnswers {
    private val TIME = Regex("^(?:what(?:'s| is) the time|what time is it|(?:tell me )?the time|time)(?: now| please)?\\??$")
    private val DATE = Regex("^(?:what(?:'s| is) (?:the date|today'?s date|the day)|what day is it|what(?:'s| is) today|(?:the )?date)(?: today| please)?\\??$")
    private val BATTERY = Regex("^(?:(?:what(?:'s| is) (?:my |the )?)?battery(?: level| left)?|how much battery(?: do i have| is left)?)\\??$")

    fun answer(context: Context, text: String): String? =
        answerFor(text, ZonedDateTime.now()) { battery(context) }

    fun answerFor(text: String, now: ZonedDateTime, battery: () -> Int?): String? {
        val q = text.trim().lowercase(Locale.UK).replace(Regex("^(?:orion[, ]+|hey orion[, ]+)"), "")
            .replace(Regex("[.!]+$"), "").replace(Regex("\\s+"), " ")
        if (q.length > 60) return null
        return when {
            TIME.matches(q) -> "It's ${now.format(DateTimeFormatter.ofPattern("HH:mm", Locale.UK))}."
            DATE.matches(q) -> "It's ${now.format(DateTimeFormatter.ofPattern("EEEE d MMMM yyyy", Locale.UK))}."
            BATTERY.matches(q) -> battery()?.let { "The battery is at $it%." }
                ?: "I can't read the battery right now."
            else -> null
        }
    }

    private fun battery(context: Context): Int? = try {
        val status = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = status?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = status?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        if (level >= 0 && scale > 0) level * 100 / scale else null
    } catch (_: Exception) { null }
}
