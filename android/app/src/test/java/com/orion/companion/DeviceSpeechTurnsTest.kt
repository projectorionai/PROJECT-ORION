package com.orion.companion

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.TimeUnit

class DeviceSpeechTurnsTest {
    @Test fun microphoneGateStaysHeldUntilTheSpeechEngineCompletes() {
        val turns = DeviceSpeechTurns()
        val turn = DeviceSpeechTurn("one")
        turns.begin(turn)
        assertFalse(turn.done.await(20, TimeUnit.MILLISECONDS))
        assertSame(turn, turns.current)
        turns.complete("one", true)
        assertTrue(turn.done.await(20, TimeUnit.MILLISECONDS))
        assertTrue(turn.success)
        // The owner retains the gate until it has consumed completion.
        assertSame(turn, turns.current)
        turns.clear(turn)
        assertNull(turns.current)
    }

    @Test fun cancellationUnblocksTheOldOwnerAndLateCallbacksCannotReleaseANewTurn() {
        val turns = DeviceSpeechTurns()
        val old = DeviceSpeechTurn("old")
        val next = DeviceSpeechTurn("new")
        turns.begin(old)
        turns.cancel()
        assertTrue(old.done.await(20, TimeUnit.MILLISECONDS))
        assertFalse(old.success)
        turns.begin(next)
        turns.complete("old", true)
        turns.clear(old)
        assertFalse(next.done.await(20, TimeUnit.MILLISECONDS))
        assertSame(next, turns.current)
        turns.complete("new", true)
        assertTrue(next.success)
    }
}
