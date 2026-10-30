package com.orion.companion

/**
 * Requests made while the phone had no internet, sent in order when it comes
 * back. Held in memory only (private words never touch storage for this),
 * bounded, and a request older than [MAX_AGE_MS] is dropped rather than
 * acted on long after it was asked.
 */
internal class OfflineQueue(private val clock: () -> Long = System::currentTimeMillis) {
    companion object {
        const val CAPACITY = 5
        const val MAX_AGE_MS = 30 * 60_000L
    }

    private val items = ArrayDeque<Pair<Long, String>>()

    /** False when full: the request is refused, never an older one evicted. */
    @Synchronized fun add(text: String): Boolean {
        expire()
        val value = text.trim()
        if (value.isEmpty() || items.size >= CAPACITY) return false
        items.addLast(clock() to value)
        return true
    }

    @Synchronized fun poll(): String? {
        expire()
        return items.removeFirstOrNull()?.second
    }

    val size: Int
        @Synchronized get() { expire(); return items.size }
    fun isNotEmpty(): Boolean = size > 0
    @Synchronized fun clear() = items.clear()

    private fun expire() {
        val now = clock()
        while (items.isNotEmpty() && now - items.first().first > MAX_AGE_MS) items.removeFirst()
    }
}
