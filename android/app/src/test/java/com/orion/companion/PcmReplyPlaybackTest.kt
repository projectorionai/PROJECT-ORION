package com.orion.companion

import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import org.junit.Assert.*
import org.junit.Test

class PcmReplyPlaybackTest {
    @Test fun fetchesLaterSentencesWhilstTheFirstOneIsStillPlaying() {
        val writing = CountDownLatch(1)
        val continuePlayback = CountDownLatch(1)
        val played = Collections.synchronizedList(mutableListOf<Byte>())
        var drains = 0
        var releases = 0
        val output = PcmReplyPlayback({ false }, { pcm ->
            writing.countDown()
            check(continuePlayback.await(2, TimeUnit.SECONDS))
            played.addAll(pcm.toList())
        }, { drains++ }, { releases++ })
        try {
            output.append(byteArrayOf(1, 2))
            assertTrue(writing.await(1, TimeUnit.SECONDS))
            // This must return while the device is blocked on the first section.
            output.append(byteArrayOf(3, 4))
            output.append(byteArrayOf(5, 6))
            assertTrue(output.hasAudio)
            assertEquals(0, drains)
            continuePlayback.countDown()
            output.finish()
            assertEquals(listOf<Byte>(1, 2, 3, 4, 5, 6), played.toList())
            assertEquals(1, drains)
            assertEquals(1, releases)
        } finally { continuePlayback.countDown(); output.close() }
    }

    @Test fun retainsTheLastSamplesOfALongSectionAcrossQueuePackets() {
        val reply = ByteArray(100_000) { (it % 127).toByte() }
        val played = Collections.synchronizedList(mutableListOf<ByteArray>())
        val output = PcmReplyPlayback({ false }, { played.add(it) }, {}, {})
        try {
            output.append(reply)
            output.finish()
            assertTrue(played.all { it.size <= 32_768 && it.size % 2 == 0 })
            assertArrayEquals(reply, played.flatMap { it.toList() }.toByteArray())
        } finally { output.close() }
    }

    @Test fun failedPlaybackNeverSilentlyCompletesTheReply() {
        val output = PcmReplyPlayback({ false }, { error("Device disconnected") }, {}, {})
        try {
            output.append(byteArrayOf(1, 2))
            try { output.finish(); fail("A failed device must be reported") }
            catch (_: IllegalStateException) { /* Expected, no fallback replay. */ }
        } finally { output.close() }
    }

    @Test fun cancellationDiscardsWaitingAudioAndDoesNotDrainOldSpeech() {
        val writing = CountDownLatch(1)
        val releaseWrite = CountDownLatch(1)
        val released = CountDownLatch(1)
        val cancelled = AtomicBoolean()
        var writes = 0
        var drained = false
        val output = PcmReplyPlayback({ cancelled.get() }, {
            writes++
            writing.countDown()
            releaseWrite.await(2, TimeUnit.SECONDS)
        }, { drained = true }, { released.countDown() })
        try {
            output.append(byteArrayOf(1, 2))
            assertTrue(writing.await(1, TimeUnit.SECONDS))
            output.append(byteArrayOf(3, 4))
            cancelled.set(true)
            releaseWrite.countDown()
            assertTrue(released.await(1, TimeUnit.SECONDS))
            assertEquals(1, writes)
            assertFalse(drained)
        } finally { releaseWrite.countDown(); output.close() }
    }
}
