package com.orion.companion

import kotlin.math.sqrt

/** A cheap, bounded level estimate for the audio actually sent to the speaker. */
internal object PcmAudioLevel {
    fun of(bytes: ByteArray, offset: Int, count: Int): Float {
        if (offset < 0 || count < 2 || offset > bytes.size - count) return 0f
        val end = offset + count - 1
        var power = 0.0
        var samples = 0
        var i = offset
        while (i < end) {
            val sample = ((bytes[i + 1].toInt() shl 8) or (bytes[i].toInt() and 0xff)).toShort().toInt()
            val normalised = sample / 32768.0
            power += normalised * normalised
            samples++
            i += 8 // Sample every fourth 16-bit value to avoid taxing playback.
        }
        return if (samples == 0) 0f else (sqrt(power / samples) * 3.5).toFloat().coerceIn(0f, 1f)
    }
}
