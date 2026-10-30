package com.orion.companion

/** Playback interruptions may stop a reply; they must never become a phone action. */
internal object BackgroundCapturePolicy {
    enum class Mode { PAUSED, REQUEST, STOP_ONLY }

    fun mode(processing: Boolean, speaking: Boolean, echoCancellation: Boolean,
             configurationChanging: Boolean, coolingDown: Boolean,
             navigationBlocked: Boolean): Mode = when {
        configurationChanging || coolingDown || navigationBlocked -> Mode.PAUSED
        speaking && echoCancellation -> Mode.STOP_ONLY
        speaking || processing -> Mode.PAUSED
        else -> Mode.REQUEST
    }
}
