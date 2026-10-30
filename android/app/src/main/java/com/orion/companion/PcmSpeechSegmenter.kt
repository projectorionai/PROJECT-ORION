package com.orion.companion

import java.io.ByteArrayOutputStream
import kotlin.math.sqrt

/** Bounded PCM16 speech windows, including the beginning of a quiet wake word. */
class PcmSpeechSegmenter {
    private val audio = ByteArrayOutputStream(32_768)
    private val leading = java.util.ArrayDeque<ByteArray>()
    private var voicedSamples = 0
    private var quietSamples = 0
    private var noiseFloor = 0.001
    var discarding = false
        private set
    var level = 0.0
        private set

    fun reset() {
        audio.reset()
        leading.clear()
        voicedSamples = 0
        quietSamples = 0
        discarding = false
        level = 0.0
    }

    fun accept(frame: ByteArray, count: Int): ByteArray? {
        require(count in 0..frame.size && count % 2 == 0)
        if (count == 0) return null
        var energy = 0.0
        for (i in 0 until count step 2) {
            val sample = ((frame[i].toInt() and 255) or (frame[i + 1].toInt() shl 8))
                .toShort().toDouble() / 32768.0
            energy += sample * sample
        }
        val samples = count / 2
        level = sqrt(energy / samples)
        // Use the same minimum as foreground dictation. Mean amplitude and a
        // 400 ms minimum previously discarded quiet speech and short wake words.
        val voice = level > maxOf(0.006, minOf(0.012, noiseFloor * 2.2))
        if (discarding) {
            quietSamples = if (voice) 0 else quietSamples + samples
            if (quietSamples >= SpeechCaptureTiming.END_SILENCE_SAMPLES) reset()
            return null // Never turn the tail of an over-budget utterance into a new command.
        }
        if (voicedSamples == 0 && !voice) {
            noiseFloor = noiseFloor * 0.98 + level * 0.02
            leading.addLast(frame.copyOf(count))
            while (leading.size > 20) leading.removeFirst() // 400 ms pre-roll.
            return null
        }
        if (voicedSamples == 0) {
            for (chunk in leading) audio.write(chunk)
            leading.clear()
        }
        if (voice) { voicedSamples += samples; quietSamples = 0 }
        else quietSamples += samples
        if (audio.size() + count > SpeechCaptureTiming.MAX_PCM_BYTES) {
            audio.reset()
            leading.clear()
            voicedSamples = 0
            discarding = true
            return null
        }
        audio.write(frame, 0, count)
        if (quietSamples >= SpeechCaptureTiming.END_SILENCE_SAMPLES) {
            val result = if (voicedSamples >= 1920) audio.toByteArray() else null // 120 ms.
            reset()
            return result
        }
        return null
    }
}
