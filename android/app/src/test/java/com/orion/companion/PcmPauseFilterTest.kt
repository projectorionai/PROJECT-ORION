package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class PcmPauseFilterTest {
    private fun pcm(milliseconds: Int, value: Int): ByteArray = ByteArray(milliseconds * 48).also { bytes ->
        for (offset in bytes.indices step 2) {
            bytes[offset] = value.toByte()
            bytes[offset + 1] = (value shr 8).toByte()
        }
    }

    @Test fun keepsNaturalBreathsAndEveryAudibleSample() {
        val audio = pcm(40, 900) + pcm(180, 0) + pcm(40, -700)
        val filter = PcmPauseFilter()
        assertArrayEquals(audio, filter.add(audio) + filter.finish())
        assertEquals(0L, filter.removedSamples)
    }

    @Test fun capsALongGeneratedPauseAcrossRequestsWithoutDroppingSpeech() {
        val filter = PcmPauseFilter()
        val first = filter.add(pcm(40, 900) + pcm(750, 0))
        val second = filter.add(pcm(750, 0) + pcm(40, -700))
        assertArrayEquals(pcm(40, 900) + pcm(260, 0) + pcm(40, -700), first + second + filter.finish())
        assertEquals(1_240L * 24, filter.removedSamples)
    }

    @Test fun preservesQuietSpeechAndPartialFinalFramesRegardlessOfFragmentation() {
        val audio = pcm(13, 700) + pcm(900, 0) + pcm(17, -700) + byteArrayOf(3, 0)
        val whole = PcmPauseFilter()
        val expected = whole.add(audio) + whole.finish()
        val fragmented = PcmPauseFilter()
        var actual = ByteArray(0)
        audio.toList().chunked(146).forEach { actual += fragmented.add(it.toByteArray()) }
        actual += fragmented.finish()
        assertArrayEquals(expected, actual)
        assertEquals(whole.removedSamples, fragmented.removedSamples)
        assertArrayEquals(byteArrayOf(3, 0), actual.takeLast(2).toByteArray())
        assertTrue(fragmented.finish().isEmpty())
    }
}
