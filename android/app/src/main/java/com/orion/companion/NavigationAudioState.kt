package com.orion.companion

/** Prevents local directions and their loudspeaker tail becoming voice commands. */
internal object NavigationAudioState {
    private val lock = Any()
    private var sequence = 0L
    private var tailUntilMillis = 0L

    @Volatile var speaking: Boolean = false
        private set

    /** A lease prevents an older speech callback from clearing a newer direction. */
    fun begin(): Long = synchronized(lock) {
        speaking = true
        ++sequence
    }

    fun end(lease: Long, nowMillis: Long) = synchronized(lock) {
        if (lease == sequence && speaking) {
            speaking = false
            tailUntilMillis = maxOf(tailUntilMillis, nowMillis + TAIL_MILLIS)
        }
    }

    fun clear(nowMillis: Long) = synchronized(lock) {
        if (speaking) tailUntilMillis = maxOf(tailUntilMillis, nowMillis + TAIL_MILLIS)
        speaking = false
        ++sequence
    }

    fun captureBlocked(nowMillis: Long): Boolean = synchronized(lock) {
        speaking || nowMillis < tailUntilMillis
    }

    private const val TAIL_MILLIS = 3_000L
}
