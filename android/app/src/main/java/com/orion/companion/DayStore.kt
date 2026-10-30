package com.orion.companion

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Workouts, the running focus block, the task in steps and the focus history,
 * kept on the phone and encrypted with the same Keystore key as the
 * conversation (workouts are health data). One small document, bounded.
 */
internal object DayStore {
    private const val NAME = "phone_day"
    const val MAX_WORKOUTS = 1000
    const val MAX_FOCUS_DAYS = 400

    @Synchronized fun read(context: Context): JSONObject =
        runCatching { JSONObject(PhoneSecretStore.get(context, NAME) ?: "{}") }.getOrDefault(JSONObject())

    @Synchronized fun update(context: Context, change: (JSONObject) -> Unit): JSONObject {
        val doc = read(context)
        change(doc)
        check(PhoneSecretStore.put(context, NAME, doc.toString())) { "The phone could not save this." }
        return doc
    }

    fun workouts(doc: JSONObject): List<DayPlanner.Workout> {
        val array = doc.optJSONArray("workouts") ?: return emptyList()
        return (0 until array.length()).mapNotNull { array.optJSONObject(it)?.let(DayPlanner.Workout::fromJson) }
            .sortedByDescending { it.at }
    }

    fun addWorkout(context: Context, kind: String, minutes: Double, distanceKm: Double?, note: String,
                   now: Long = System.currentTimeMillis()): DayPlanner.Workout {
        DayPlanner.checkWorkout(minutes, distanceKm)?.let { throw IllegalArgumentException(it) }
        val workout = DayPlanner.Workout(now, DayPlanner.normaliseKind(kind), Math.round(minutes * 10) / 10.0,
            distanceKm?.let { Math.round(it * 100) / 100.0 }, now, note.trim().take(200))
        update(context) { doc ->
            val kept = (listOf(workout) + workouts(doc)).take(MAX_WORKOUTS)
            doc.put("workouts", JSONArray().apply { kept.forEach { put(it.toJson()) } })
        }
        return workout
    }

    fun removeWorkout(context: Context, id: Long) {
        update(context) { doc ->
            doc.put("workouts", JSONArray().apply { workouts(doc).filter { it.id != id }.forEach { put(it.toJson()) } })
        }
    }

    fun target(doc: JSONObject): Int = doc.optInt("weekly_target", DayPlanner.DEFAULT_WEEKLY_WORKOUTS).coerceIn(1, 14)

    fun focusHistory(doc: JSONObject): List<Pair<Long, Int>> {
        val array = doc.optJSONArray("focus") ?: return emptyList()
        return (0 until array.length()).mapNotNull { i ->
            array.optJSONObject(i)?.let { it.optLong("at") to it.optInt("minutes") }?.takeIf { it.second > 0 }
        }
    }

    /** A finished block: when it ended and how many minutes it held. */
    fun recordFocus(doc: JSONObject, at: Long, minutes: Int) {
        if (minutes <= 0) return
        val cutoff = at - MAX_FOCUS_DAYS * 86_400_000L
        val kept = focusHistory(doc).filter { it.first >= cutoff } + (at to minutes)
        doc.put("focus", JSONArray().apply { kept.forEach { put(JSONObject().put("at", it.first).put("minutes", it.second)) } })
    }
}
