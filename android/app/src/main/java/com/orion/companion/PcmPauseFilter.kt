package com.orion.companion

import java.io.ByteArrayOutputStream

/** Shorten sustained near-silence, retaining natural pauses and every audible frame. */
internal class PcmPauseFilter {
    private var carry = ByteArray(0)
    private var silentFrames = 0
    private var ended = false
    var removedSamples = 0L
        private set

    fun add(pcm: ByteArray): ByteArray {
        require(pcm.size % 2 == 0)
        check(!ended)
        val input = carry + pcm
        val output = ByteArrayOutputStream(input.size)
        var offset = 0
        // Ten milliseconds at 24 kHz. A low peak threshold protects quiet speech.
        while (offset + 480 <= input.size) {
            var quiet = true
            for (sample in offset until offset + 480 step 2) {
                val value = ((input[sample].toInt() and 0xff) or
                    (input[sample + 1].toInt() shl 8)).toShort().toInt()
                if (kotlin.math.abs(value) > 120) { quiet = false; break }
            }
            if (quiet) silentFrames++ else silentFrames = 0
            // Preserve up to 260 ms; this is not a voice activity detector.
            if (silentFrames <= 26) output.write(input, offset, 480)
            else removedSamples += 240
            offset += 480
        }
        carry = input.copyOfRange(offset, input.size)
        return output.toByteArray()
    }

    fun finish(): ByteArray {
        if (ended) return ByteArray(0)
        ended = true
        return carry.also { carry = ByteArray(0) }
    }
}
