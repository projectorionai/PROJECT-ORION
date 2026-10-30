package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiveTurnGuardTest {
    @Test fun failedSetupOrEmptyTurnMayUseExistingCharonStream() {
        val failedSetup = LiveTurnGuard()
        assertEquals(LiveSpeechResult.UNAVAILABLE, failedSetup.failed())
        assertEquals(LiveSpeechResult.UNAVAILABLE, LiveTurnGuard().completed())
        assertFalse(failedSetup.audioMayHavePlayed)
    }

    @Test fun interruptedOrFailedPlaybackMustNeverReplayTheAnswer() {
        val interrupted = LiveTurnGuard()
        interrupted.beforePlayback()
        assertEquals(LiveSpeechResult.PARTIAL, interrupted.failed())
        assertTrue(interrupted.audioMayHavePlayed)
        assertFalse(interrupted.finishedSuccessfully)
    }

    @Test fun completedTurnIsKeptOnlyAfterAudioArrives() {
        val spoken = LiveTurnGuard()
        spoken.beforePlayback()
        assertEquals(LiveSpeechResult.COMPLETE, spoken.completed())
        assertTrue(spoken.finishedSuccessfully)
        assertEquals(LiveSpeechResult.CANCELLED, spoken.failed(cancelled = true))
    }

    @Test fun connectionBackoffAvoidsRapidRetriesAndSeparatesQuotaFromAuthFailure() {
        assertEquals(30_000L, LiveReconnectPolicy.pauseMillis(null))
        assertEquals(30_000L, LiveReconnectPolicy.pauseMillis(503))
        assertEquals(180_000L, LiveReconnectPolicy.pauseMillis(429))
        assertEquals(86_400_000L, LiveReconnectPolicy.pauseMillis(403))
    }
}
