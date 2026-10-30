package com.orion.companion

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class PhoneMediaProtocolTest {
    private fun events(text: String, limit: Int = 2_097_152) =
        PhoneSseEvents(ByteArrayInputStream(text.toByteArray()), limit)

    @Test fun speechEventsHandleCommentsMultilineAndCrLf() {
        val source = events("\uFEFF: ping\r\nevent: message\r\ndata: {\r\ndata: \"done\":true}\r\n\r\ndata: final\n\n")
        assertEquals("{\n\"done\":true}", source.next())
        assertEquals("final", source.next())
        assertNull(source.next())
    }
    @Test fun speechEventsRejectTruncationAndOversizedLines() {
        assertThrows(IllegalStateException::class.java) { events("data: incomplete\n").next() }
        assertThrows(IllegalStateException::class.java) { events("data: missing separator").next() }
        assertThrows(IllegalStateException::class.java) { events("data: " + "x".repeat(40) + "\n\n", 32).next() }
    }
    @Test fun backgroundCommandsRequireWakePrefixAndExactInterruption() {
        assertEquals("open my email", PhoneWakeWords.request("Hey ORION, open my email"))
        assertEquals("", PhoneWakeWords.request("ORION"))
        assertEquals("hello.", PhoneWakeWords.request("‘Orion, hello.’"))
        assertEquals("show my route", PhoneWakeWords.request("Hey O.R.I.O.N., show my route"))
        assertEquals("open my messages", PhoneWakeWords.request("Orian open my messages"))
        assertNull(PhoneWakeWords.request("My television said ORION open email"))
        assertNull(PhoneWakeWords.request("orions stop"))
        assertTrue(PhoneWakeWords.isStop("Okay ORION, stop speaking!"))
        assertFalse(PhoneWakeWords.isStop("ORION stop and send an email"))
        assertTrue(PhoneWakeWords.stopListening("stop listening."))
        assertFalse(PhoneWakeWords.stopListening("stop listening and send the draft"))
    }
    @Test fun continuousListeningAcceptsNaturalSpeechAndOnlyExactStopCommands() {
        assertEquals("open Calculator", PhoneWakeWords.conversationRequest("open Calculator"))
        assertEquals("open Calculator", PhoneWakeWords.conversationRequest("ORION, open Calculator"))
        assertEquals("show my route", PhoneWakeWords.conversationRequest("Hey ORION, show my route"))
        assertTrue(PhoneWakeWords.isConversationStop("stop speaking"))
        assertTrue(PhoneWakeWords.isConversationStop("ORION stop"))
        assertFalse(PhoneWakeWords.isConversationStop("stop speaking and open email"))
        assertTrue(PhoneWakeWords.stopListening(PhoneWakeWords.conversationRequest("stop listening")))
    }
    @Test fun retryDelayIsBoundedAndWavKeepsEverySample() {
        assertEquals(2_000L, PhoneWakeWords.retryDelay(1))
        assertEquals(60_000L, PhoneWakeWords.retryDelay(99))
        val pcm = ByteArray(640) { it.toByte() }
        val wave = PhoneWakeWords.wav(pcm)
        assertEquals("RIFF", String(wave, 0, 4, Charsets.US_ASCII))
        assertEquals(16_000, ByteBuffer.wrap(wave, 24, 4).order(ByteOrder.LITTLE_ENDIAN).int)
        assertArrayEquals(pcm, wave.copyOfRange(44, wave.size))
        assertThrows(IllegalArgumentException::class.java) { PhoneWakeWords.wav(ByteArray(3)) }
    }
    @Test fun fullThirtySecondWavIsBoundedAndRetainsItsFinalSamples() {
        val pcm = ByteArray(SpeechCaptureTiming.MAX_PCM_BYTES) { (it % 127).toByte() }
        val wave = PhoneWakeWords.wav(pcm)
        assertEquals(960_044, wave.size)
        assertEquals(pcm.size, ByteBuffer.wrap(wave, 40, 4).order(ByteOrder.LITTLE_ENDIAN).int)
        assertArrayEquals(pcm, wave.copyOfRange(44, wave.size))
        assertThrows(IllegalArgumentException::class.java) {
            PhoneWakeWords.wav(ByteArray(SpeechCaptureTiming.MAX_PCM_BYTES + 2))
        }
    }
    @Test fun screenEvidenceExpiresCopiesAndClears() {
        val cache = PhoneFrameCache(maxBytes = 16)
        val jpeg = byteArrayOf(0xff.toByte(), 0xd8.toByte(), 0, 0xff.toByte(), 0xd9.toByte())
        cache.put(jpeg, 100)
        jpeg[2] = 5
        assertEquals(0, cache.get(200)!![2].toInt())
        cache.get(200)!![2] = 9
        assertEquals(0, cache.get(201)!![2].toInt())
        assertNull(cache.get(10_100))
        cache.put(ByteArray(17), 11_000)
        assertNull(cache.get(11_001))
        cache.put(jpeg, 12_000); cache.clear()
        assertNull(cache.get(12_001))
    }
}
