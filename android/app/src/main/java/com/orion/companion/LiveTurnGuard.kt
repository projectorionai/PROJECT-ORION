package com.orion.companion

/** Prevents a second speech engine from repeating audio already played. */
internal enum class LiveSpeechResult { COMPLETE, UNAVAILABLE, PARTIAL, CANCELLED }

internal class LiveTurnGuard {
    var audioMayHavePlayed = false
        private set
    var finishedSuccessfully = false
        private set

    fun beforePlayback() { audioMayHavePlayed = true }

    fun completed(): LiveSpeechResult {
        finishedSuccessfully = audioMayHavePlayed
        return if (finishedSuccessfully) LiveSpeechResult.COMPLETE else LiveSpeechResult.UNAVAILABLE
    }

    fun failed(cancelled: Boolean = false): LiveSpeechResult = when {
        cancelled -> LiveSpeechResult.CANCELLED
        audioMayHavePlayed -> LiveSpeechResult.PARTIAL
        else -> LiveSpeechResult.UNAVAILABLE
    }
}

internal object LiveReconnectPolicy {
    fun pauseMillis(httpStatus: Int?): Long = when (httpStatus) {
        400, 401, 403, 404 -> java.util.concurrent.TimeUnit.HOURS.toMillis(24)
        429 -> java.util.concurrent.TimeUnit.MINUTES.toMillis(3)
        else -> 30_000L
    }
}
