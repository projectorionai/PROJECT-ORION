package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class SpeechPlaybackEpochTest {
    @Test fun anOldStopCannotApplyToTheNextForegroundOrBackgroundReply() {
        val recordedOver = SpeechPlaybackEpoch.begin()
        assertTrue(SpeechPlaybackEpoch.isCurrent(recordedOver))
        val nextReply = SpeechPlaybackEpoch.begin()
        assertFalse(SpeechPlaybackEpoch.isCurrent(recordedOver))
        assertTrue(SpeechPlaybackEpoch.isCurrent(nextReply))
        assertEquals(nextReply, SpeechPlaybackEpoch.current())
    }
}
