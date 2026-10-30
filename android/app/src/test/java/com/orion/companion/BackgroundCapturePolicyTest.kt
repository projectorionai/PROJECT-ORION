package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Test

class BackgroundCapturePolicyTest {
    private fun mode(processing: Boolean = false, speaking: Boolean = false,
                     echo: Boolean = true, navigation: Boolean = false,
                     changing: Boolean = false, cooldown: Boolean = false) =
        BackgroundCapturePolicy.mode(processing, speaking, echo, changing, cooldown, navigation)

    @Test fun aReplyStillStreamingCanBeStoppedButCannotAcceptAnotherAction() {
        assertEquals(BackgroundCapturePolicy.Mode.STOP_ONLY, mode(processing = true, speaking = true))
        assertEquals(BackgroundCapturePolicy.Mode.PAUSED, mode(processing = true))
        assertEquals(BackgroundCapturePolicy.Mode.REQUEST, mode())
    }

    @Test fun navigationAndItsAcousticTailNeverEnterTheTranscriptionPipeline() {
        for (working in listOf(false, true)) for (talking in listOf(false, true)) {
            assertEquals(BackgroundCapturePolicy.Mode.PAUSED,
                mode(processing = working, speaking = talking, navigation = true))
        }
    }

    @Test fun missingEchoCancellationAndConfigurationChangesKeepPlaybackClosed() {
        assertEquals(BackgroundCapturePolicy.Mode.PAUSED, mode(speaking = true, echo = false))
        assertEquals(BackgroundCapturePolicy.Mode.PAUSED, mode(speaking = true, changing = true))
        assertEquals(BackgroundCapturePolicy.Mode.PAUSED, mode(speaking = true, cooldown = true))
    }
}
