package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class SpeechCaptureTimingTest {
    @Test fun waitsForTheWholeBreathAndPreservesAShortOpeningWord() {
        assertFalse(SpeechCaptureTiming.shouldFinish(100, 260, 759))
        assertFalse(SpeechCaptureTiming.shouldFinish(100, 260, 860))
        assertTrue(SpeechCaptureTiming.shouldFinish(100, 260, 900))
    }

    @Test fun continuingSpeechRestartsTheSilenceBudget() {
        assertFalse(SpeechCaptureTiming.shouldFinish(100, 1_200, 1_799))
        assertTrue(SpeechCaptureTiming.shouldFinish(100, 1_200, 1_800))
    }

    @Test fun silenceAndInvalidMonotonicTimesCannotEndCaptureAsSpeech() {
        assertFalse(SpeechCaptureTiming.shouldFinish(100, 0, 20_000))
        assertFalse(SpeechCaptureTiming.shouldFinish(100, 800, 799))
        assertFalse(SpeechCaptureTiming.shouldFinish(2_000, 800, 1_999))
    }
}
