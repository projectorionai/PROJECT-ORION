package com.orion.companion

/** Shared end-of-speech timing for foreground dictation and continuous listening. */
internal object SpeechCaptureTiming {
    const val SAMPLE_RATE = 16_000
    const val END_SILENCE_MILLIS = 600L
    const val END_SILENCE_SAMPLES = SAMPLE_RATE * 600 / 1_000
    const val MAX_UTTERANCE_MILLIS = 30_000L
    const val MAX_PCM_BYTES = SAMPLE_RATE * 2 * 30
    private const val MINIMUM_CAPTURE_MILLIS = 800L

    /** Monotonic capture times only; a brief breath does not submit a partial request. */
    fun shouldFinish(startedAt: Long, lastVoiceAt: Long, now: Long): Boolean =
        lastVoiceAt > 0L && now >= lastVoiceAt && now >= startedAt &&
            now - lastVoiceAt >= END_SILENCE_MILLIS &&
            now - startedAt >= MINIMUM_CAPTURE_MILLIS
}
