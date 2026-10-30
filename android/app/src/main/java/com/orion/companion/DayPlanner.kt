package com.orion.companion

import org.json.JSONArray
import org.json.JSONObject
import java.time.DayOfWeek
import java.time.Duration
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.ZonedDateTime
import java.time.format.TextStyle
import java.util.Locale

/**
 * The phone's own day: focus blocks, the task in small steps, workouts and the
 * week ahead. The same rules as the PC's Focus, Fitness and Calendar pages
 * (orion_core/focus.py, fitness.py, gui/calendar_deck.py), so the two read
 * alike. No Android types here - the screens and the store sit on top.
 */
internal object DayPlanner {

    // ── focus ──────────────────────────────────────────────────────────────

    data class Mode(val key: String, val name: String, val minutes: Int, val bodyDouble: Boolean = false)

    val MODES = listOf(Mode("15", "15 min", 15), Mode("25", "25 min", 25), Mode("45", "45 min", 45),
        Mode("body", "Body double", 25, bodyDouble = true))

    data class Block(val label: String, val startedAt: Long, val minutes: Int, val breakMinutes: Int,
                     val bodyDouble: Boolean) {
        fun elapsedMs(now: Long) = (now - startedAt).coerceAtLeast(0)
        fun totalMs() = minutes * 60_000L
        fun toJson(): JSONObject = JSONObject().put("label", label).put("started", startedAt)
            .put("minutes", minutes).put("break", breakMinutes).put("body", bodyDouble)

        companion object {
            fun fromJson(json: JSONObject?): Block? = json?.let {
                val minutes = it.optInt("minutes", 0)
                if (minutes <= 0) null else Block(it.optString("label"), it.optLong("started"), minutes,
                    it.optInt("break", breakFor(minutes)), it.optBoolean("body"))
            }
        }
    }

    /** A sensible break scales with the block: a fifth of it, at least three minutes. */
    fun breakFor(minutes: Int): Int = maxOf(3, Math.round(minutes / 5.0).toInt())

    data class Dial(val mode: String, val leftMs: Long, val progress: Float, val text: String,
                    val caption: String)

    fun clockText(ms: Long): String {
        val seconds = ((ms.coerceAtLeast(0) + 500) / 1000)
        return "%02d:%02d".format(seconds / 60, seconds % 60)
    }

    fun dial(block: Block?, now: Long, idleMinutes: Int, breakUntil: Long = 0, breakMs: Long = 0,
             caption: String = ""): Dial {
        if (block == null && breakUntil > now) {
            val left = breakUntil - now
            val total = breakMs.coerceAtLeast(1)
            return Dial("break", left, 1f - left.toFloat() / total, clockText(left), "Break")
        }
        if (block == null) {
            val total = idleMinutes * 60_000L
            return Dial("idle", total, 0f, clockText(total), caption)
        }
        val left = (block.totalMs() - block.elapsedMs(now)).coerceAtLeast(0)
        if (left == 0L) return Dial("due", 0, 1f, "00:00", "Time for a short break")
        return Dial("running", left, 1f - left.toFloat() / block.totalMs(), clockText(left),
            caption.ifBlank { block.label })
    }

    /** A block stopped past its halfway point still counts as done. */
    fun gentleStopCounts(block: Block?, now: Long): Boolean =
        block != null && block.elapsedMs(now) * 2 >= block.totalMs()

    private val STEP_BREAKS = Regex("\\n|;|•|,|\\band then\\b|\\bthen\\b|\\bafter that\\b", RegexOption.IGNORE_CASE)
    private val STEP_LEAD = Regex("^\\s*(?:[-*]\\s*|\\d+[.)]\\s*|and\\s+|then\\s+)+", RegexOption.IGNORE_CASE)
    private val AND = Regex("\\s+and\\s+", RegexOption.IGNORE_CASE)

    /** "outline the essay, write the intro then add two sources" -> three steps;
     *  "buy salt and pepper" stays one. Only the person's own words are used. */
    fun chunkTask(text: String, limit: Int = 8): List<String> {
        val steps = mutableListOf<String>()
        val seen = mutableSetOf<String>()
        for (part in STEP_BREAKS.split(text)) {
            val halves = AND.split(part.trim(), 2)
            val pieces = if (halves.size == 2 && halves.all { it.trim().split(Regex("\\s+")).size >= 2 })
                halves else listOf(part)
            for (piece in pieces) {
                val step = STEP_LEAD.replace(piece, "").trim(' ', '.')
                if (step.isEmpty() || !seen.add(step.lowercase(Locale.UK))) continue
                steps += step.replaceFirstChar { it.uppercase(Locale.UK) }
                if (steps.size >= limit) return steps
            }
        }
        return steps
    }

    data class Step(val text: String, val done: Boolean)

    data class FocusDay(val minutes: Int, val blocks: Int, val streak: Int)

    /** Today's focused minutes and finished blocks, and the run of days with a
     *  finished block (today not yet focused does not break it). */
    fun focusDay(history: List<Pair<Long, Int>>, now: ZonedDateTime): FocusDay {
        val zone = now.zone
        val today = now.toLocalDate()
        val todays = history.filter { localDay(it.first, zone) == today }
        val days = history.map { localDay(it.first, zone) }.toSet()
        return FocusDay(todays.sumOf { it.second }, todays.size, streak(days, today))
    }

    // ── fitness ────────────────────────────────────────────────────────────

    val KINDS = listOf("run", "walk", "cycle", "strength", "yoga", "swim", "sport", "other")
    val DISTANCE_KINDS = setOf("run", "walk", "cycle", "swim")
    const val DEFAULT_WEEKLY_WORKOUTS = 3

    fun kindName(kind: String) = when (kind) {
        "cycle" -> "Cycle"; "strength" -> "Strength"; else -> kind.replaceFirstChar { it.uppercase(Locale.UK) }
    }

    fun normaliseKind(kind: String): String {
        val word = kind.trim().lowercase(Locale.UK)
        val aliased = mapOf("running" to "run", "jog" to "run", "jogging" to "run", "walking" to "walk",
            "hike" to "walk", "cycling" to "cycle", "bike" to "cycle", "ride" to "cycle",
            "gym" to "strength", "weights" to "strength", "lifting" to "strength",
            "swimming" to "swim", "stretch" to "yoga", "pilates" to "yoga")[word] ?: word
        return if (aliased in KINDS) aliased else "other"
    }

    data class Workout(val id: Long, val kind: String, val minutes: Double, val distanceKm: Double?,
                       val at: Long, val note: String = "") {
        fun toJson(): JSONObject = JSONObject().put("id", id).put("kind", kind).put("minutes", minutes)
            .put("km", distanceKm ?: JSONObject.NULL).put("at", at).put("note", note)

        companion object {
            fun fromJson(json: JSONObject): Workout? {
                val minutes = json.optDouble("minutes", 0.0)
                if (minutes <= 0) return null
                val km = if (json.isNull("km")) null else json.optDouble("km").takeIf { it > 0 }
                return Workout(json.optLong("id"), normaliseKind(json.optString("kind")), minutes, km,
                    json.optLong("at"), json.optString("note"))
            }
        }
    }

    /** Checks a workout before it is kept; the message is what the page shows. */
    fun checkWorkout(minutes: Double, distanceKm: Double?): String? = when {
        minutes <= 0 || minutes > 24 * 60 -> "A workout lasts between a minute and a day."
        distanceKm != null && (distanceKm <= 0 || distanceKm >= 1000) -> "That distance does not look right."
        else -> null
    }

    data class Week(val start: LocalDate, val days: List<Pair<LocalDate, Double>>, val workouts: Int,
                    val minutes: Int, val distanceKm: Double, val target: Int, val streak: Int,
                    val today: LocalDate)

    fun streak(days: Set<LocalDate>, today: LocalDate): Int {
        var day = if (today in days) today else today.minusDays(1)
        var count = 0
        while (day in days) { count++; day = day.minusDays(1) }
        return count
    }

    /** This week, Monday to Sunday, in the phone's own time. */
    fun week(workouts: List<Workout>, now: ZonedDateTime, target: Int = DEFAULT_WEEKLY_WORKOUTS): Week {
        val zone = now.zone
        val today = now.toLocalDate()
        val monday = today.with(DayOfWeek.MONDAY)
        val perDay = linkedMapOf<LocalDate, Double>()
        for (i in 0L until 7L) perDay[monday.plusDays(i)] = 0.0
        val thisWeek = workouts.filter { localDay(it.at, zone) in perDay }
        thisWeek.forEach { perDay[localDay(it.at, zone)] = perDay.getValue(localDay(it.at, zone)) + it.minutes }
        return Week(monday, perDay.toList(), thisWeek.size, Math.round(thisWeek.sumOf { it.minutes }).toInt(),
            Math.round(thisWeek.sumOf { it.distanceKm ?: 0.0 } * 10) / 10.0, target,
            streak(workouts.map { localDay(it.at, zone) }.toSet(), today), today)
    }

    /** '45m', '1:05' - a week's time at a glance. */
    fun clock(minutes: Double): String {
        val m = Math.round(minutes).toInt()
        return if (m < 60) "${m}m" else "${m / 60}:%02d".format(m % 60)
    }

    fun amountText(workout: Workout): String {
        val time = "${Math.round(workout.minutes)} min"
        val km = workout.distanceKm ?: return time
        return "$time · ${"%.1f".format(Locale.UK, km).removeSuffix(".0")} km"
    }

    fun whenText(at: Long, now: ZonedDateTime): String {
        val stamp = Instant.ofEpochMilli(at).atZone(now.zone)
        val days = Duration.between(stamp.toLocalDate().atStartOfDay(now.zone),
            now.toLocalDate().atStartOfDay(now.zone)).toDays()
        return when {
            days == 0L -> "Today, %02d:%02d".format(stamp.hour, stamp.minute)
            days == 1L -> "Yesterday"
            days in 2..6 -> stamp.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.UK)
            else -> "${stamp.dayOfMonth} ${stamp.month.getDisplayName(TextStyle.SHORT, Locale.UK)}"
        }
    }

    // ── calendar ───────────────────────────────────────────────────────────

    data class Entry(val day: LocalDate, val title: String, val at: ZonedDateTime?, val source: String) {
        fun timeText(): String = at?.let { "%02d:%02d".format(it.hour, it.minute) } ?: "All day"
    }

    private fun order(e: Entry) = Triple(e.day, if (e.at != null) 0 else 1, e.at?.toEpochSecond() ?: 0L)

    fun weekOf(entries: List<Entry>, today: LocalDate, days: Int = 7): List<Pair<LocalDate, List<Entry>>> =
        (0 until days).map { i ->
            val day = today.plusDays(i.toLong())
            day to entries.filter { it.day == day }.sortedWith(compareBy({ order(it).second },
                { order(it).third }, { it.title.lowercase(Locale.UK) }))
        }

    /** The next timed entry still to come (five minutes' grace), or the soonest all-day one. */
    fun nextUp(entries: List<Entry>, now: ZonedDateTime): Entry? {
        val grace = now.minusMinutes(5)
        return entries.filter { (it.at != null && !it.at.isBefore(grace)) || (it.at == null && !it.day.isBefore(now.toLocalDate())) }
            .minWithOrNull(compareBy({ it.day }, { if (it.at != null) 0 else 1 }, { it.at?.toEpochSecond() ?: 0L }))
    }

    /** 'in 25 min', 'in 2 h 10 min', 'Tomorrow', 'Wednesday'. */
    fun relative(entry: Entry, now: ZonedDateTime): String {
        val days = Duration.between(now.toLocalDate().atStartOfDay(), entry.day.atStartOfDay()).toDays()
        if (entry.at != null && days == 0L) {
            val minutes = Duration.between(now, entry.at).toMinutes()
            return when {
                minutes <= 0 -> "Now"
                minutes < 60 -> "in $minutes min"
                minutes % 60 == 0L -> "in ${minutes / 60} h"
                else -> "in ${minutes / 60} h ${minutes % 60} min"
            }
        }
        return when (days) {
            0L -> "Today"
            1L -> "Tomorrow"
            else -> entry.day.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.UK)
        }
    }

    // ── asking for a page ──────────────────────────────────────────────────

    private val CALENDAR_ASK = Regex("^(?:open |show(?: me)? )?(?:my |the )?(?:calendar|diary|agenda|week ahead)(?: page)?$")
    private val FOCUS_ASK = Regex("^(?:open |start )?(?:my |the |a )?(?:focus(?: mode| timer| block| session| page)?|pomodoro|body doubl(?:e|ing))$")
    private val FITNESS_ASK = Regex("^(?:open |show(?: me)? )?(?:my |the )?(?:fitness(?: page)?|workouts?|training log)$|^log (?:a |my )?(?:workout|run|walk)$")

    /** Which page a short request opens ("open my calendar", "start a focus
     *  block", "log a workout"), or null for everything else. */
    fun pageFor(text: String): String? {
        val q = text.trim().lowercase(Locale.UK).replace(Regex("^(?:hey orion|orion)[, ]+"), "")
            .replace(Regex("[.!?]+$"), "").replace(Regex("\\s+"), " ")
        if (q.length > 40) return null
        return when {
            CALENDAR_ASK.matches(q) -> "calendar"
            FOCUS_ASK.matches(q) -> "focus"
            FITNESS_ASK.matches(q) -> "fitness"
            else -> null
        }
    }

    // ── shared ─────────────────────────────────────────────────────────────

    fun localDay(epochMs: Long, zone: ZoneId): LocalDate = Instant.ofEpochMilli(epochMs).atZone(zone).toLocalDate()

    fun stepsToJson(steps: List<Step>): JSONArray = JSONArray().apply {
        steps.forEach { put(JSONObject().put("text", it.text).put("done", it.done)) }
    }

    fun stepsFromJson(array: JSONArray?): List<Step> = buildList {
        if (array == null) return@buildList
        for (i in 0 until array.length()) {
            val item = array.optJSONObject(i) ?: continue
            val text = item.optString("text").trim()
            if (text.isNotEmpty()) add(Step(text.take(200), item.optBoolean("done")))
        }
    }
}
