package com.orion.companion

import java.util.concurrent.atomic.AtomicLong
import org.junit.Assert.*
import org.junit.Test

class NavigationAudioStateTest {
    companion object { private val clock = AtomicLong(1_000_000L) }
    private fun now(): Long = clock.addAndGet(10_000L).also { NavigationAudioState.clear(it) }

    @Test fun captureRemainsClosedUntilTheActualPlaybackTailExpires() {
        val start = now()
        val lease = NavigationAudioState.begin()
        assertTrue(NavigationAudioState.captureBlocked(start))
        NavigationAudioState.end(lease, start + 1_000L)
        assertFalse(NavigationAudioState.speaking)
        assertTrue(NavigationAudioState.captureBlocked(start + 3_999L))
        assertFalse(NavigationAudioState.captureBlocked(start + 4_000L))
    }

    @Test fun anOldUtteranceCallbackCannotUnblockANewerDirection() {
        val start = now()
        val old = NavigationAudioState.begin()
        val current = NavigationAudioState.begin()
        NavigationAudioState.end(old, start)
        assertTrue(NavigationAudioState.speaking)
        assertTrue(NavigationAudioState.captureBlocked(start + 5_000L))
        NavigationAudioState.end(current, start + 5_000L)
        assertFalse(NavigationAudioState.speaking)
        assertTrue(NavigationAudioState.captureBlocked(start + 7_999L))
        assertFalse(NavigationAudioState.captureBlocked(start + 8_000L))
    }

    @Test fun stoppingTheServiceInvalidatesCallbacksAndRetainsTheSpeakerTail() {
        val start = now()
        val lease = NavigationAudioState.begin()
        NavigationAudioState.clear(start + 500L)
        NavigationAudioState.end(lease, start + 900L)
        assertFalse(NavigationAudioState.speaking)
        assertTrue(NavigationAudioState.captureBlocked(start + 3_499L))
        assertFalse(NavigationAudioState.captureBlocked(start + 3_500L))
    }
}
