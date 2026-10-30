package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PcmAudioLevelTest {
    @Test fun silenceHasNoVisibleOutput() {
        assertEquals(0f, PcmAudioLevel.of(ByteArray(64), 0, 64))
    }

    @Test fun speechSamplesDriveTheVisualWithoutOverflow() {
        val samples = ByteArray(64) { if (it % 2 == 0) 0 else 64 }
        val level = PcmAudioLevel.of(samples, 0, samples.size)
        assertTrue(level > 0f && level <= 1f)
        assertEquals(0f, PcmAudioLevel.of(samples, -1, samples.size))
        assertEquals(0f, PcmAudioLevel.of(samples, 50, 20))
    }
}
