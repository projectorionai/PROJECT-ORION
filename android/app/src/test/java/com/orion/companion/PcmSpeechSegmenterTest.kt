package com.orion.companion

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.sin

class PcmSpeechSegmenterTest {
    private fun frame(amplitude: Double): ByteArray = ByteArray(640).also { bytes ->
        for (i in 0 until 320) {
            val value = (sin(i * 0.2) * amplitude * 32767).toInt()
            bytes[i * 2] = value.toByte()
            bytes[i * 2 + 1] = (value shr 8).toByte()
        }
    }

    @Test fun quietShortWakeWordSurvives() {
        val detector = PcmSpeechSegmenter()
        repeat(30) { assertNull(detector.accept(frame(0.001), 640)) }
        repeat(8) { assertNull(detector.accept(frame(0.010), 640)) }
        var result: ByteArray? = null
        repeat(30) { result = detector.accept(frame(0.001), 640) ?: result }
        assertNotNull(result)
        assertEquals((20 + 8 + 30) * 640, result!!.size)
    }

    @Test fun noiseAndSingleClickDoNotTrigger() {
        val detector = PcmSpeechSegmenter()
        repeat(100) { assertNull(detector.accept(frame(0.002), 640)) }
        assertNull(detector.accept(frame(0.8), 640))
        repeat(60) { assertNull(detector.accept(frame(0.002), 640)) }
    }

    @Test fun resetClearsLeadingAudioAndPartialSpeech() {
        val detector = PcmSpeechSegmenter()
        repeat(500) { assertNull(detector.accept(frame(0.03), 640)) }
        detector.reset()
        repeat(10) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(29) { assertNull(detector.accept(frame(0.001), 640)) }
        assertEquals(40 * 640, detector.accept(frame(0.001), 640)!!.size)
    }

    @Test fun twentySecondRequestRemainsWholeInsteadOfBeingSplitIntoCommands() {
        val detector = PcmSpeechSegmenter()
        repeat(1_000) { assertNull(detector.accept(frame(0.03), 640)) }
        assertFalse(detector.discarding)
        repeat(29) { assertNull(detector.accept(frame(0.001), 640)) }
        assertEquals(1_030 * 640, detector.accept(frame(0.001), 640)!!.size)
    }

    @Test fun overThirtySecondsDiscardsTheEntireTailUntilRealSilence() {
        val detector = PcmSpeechSegmenter()
        repeat(1_501) { assertNull(detector.accept(frame(0.03), 640)) }
        assertTrue(detector.discarding)
        repeat(100) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(25) { assertNull(detector.accept(frame(0.001), 640)) }
        repeat(10) { assertNull(detector.accept(frame(0.03), 640)) }
        assertTrue(detector.discarding) // A half-second breath is not the end of the discarded request.
        repeat(30) { assertNull(detector.accept(frame(0.001), 640)) }
        assertFalse(detector.discarding)
        repeat(8) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(29) { assertNull(detector.accept(frame(0.001), 640)) }
        assertEquals(38 * 640, detector.accept(frame(0.001), 640)!!.size)
    }

    @Test fun explicitResetCancelsDiscardingWithoutReleasingOldAudio() {
        val detector = PcmSpeechSegmenter()
        repeat(1_501) { assertNull(detector.accept(frame(0.03), 640)) }
        assertTrue(detector.discarding)
        detector.reset()
        assertFalse(detector.discarding)
        repeat(8) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(29) { assertNull(detector.accept(frame(0.001), 640)) }
        assertEquals(38 * 640, detector.accept(frame(0.001), 640)!!.size)
    }

    @Test fun halfSecondBreathKeepsBothPartsOfTheSameRequest() {
        val detector = PcmSpeechSegmenter()
        repeat(10) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(25) { assertNull(detector.accept(frame(0.001), 640)) }
        repeat(10) { assertNull(detector.accept(frame(0.03), 640)) }
        repeat(29) { assertNull(detector.accept(frame(0.001), 640)) }
        assertEquals(75 * 640, detector.accept(frame(0.001), 640)!!.size)
    }

    @Test fun endpointWaitUsesSamplesEvenWhenMicrophoneFramesArePartial() {
        val detector = PcmSpeechSegmenter()
        repeat(8) { assertNull(detector.accept(frame(0.03), 640)) }
        val quiet = frame(0.001)
        repeat(59) { assertNull(detector.accept(quiet, 320)) }
        assertEquals(8 * 640 + 60 * 320, detector.accept(quiet, 320)!!.size)
    }
}
