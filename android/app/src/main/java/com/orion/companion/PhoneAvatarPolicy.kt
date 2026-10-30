package com.orion.companion

/**
 * What ORION's Screen Face shows on the phone. One face, the PC's - the
 * Three.js orb/face modes are gone (owner, 29 Oct 2026). The words are the
 * desktop's state words, which the face maps onto its state families
 * (STANDBY rests and later dozes, LISTENING leans in, PROCESSING thinks with
 * the three dots, SPEAKING moves the mouth, OFFLINE dims).
 */
internal object PhoneAvatarPolicy {
    fun state(listening: Boolean, thinking: Boolean, speaking: Boolean, offline: Boolean = false): String = when {
        speaking -> "SPEAKING"
        thinking -> "PROCESSING"
        listening -> "LISTENING"
        offline -> "OFFLINE"
        else -> "STANDBY"
    }

    /** A state word safe to place in the page script: letters, space and
     *  underscore only, bounded; anything else rests at STANDBY. */
    fun faceState(raw: String): String = raw.uppercase()
        .filter { it in 'A'..'Z' || it == ' ' || it == '_' }.trim().take(24)
        .ifBlank { "STANDBY" }

    /** While the face follows the PC, what the phone is doing itself still
     *  wins (you are talking to the phone); otherwise it is the PC's state. */
    fun linked(pcState: String, phoneState: String): String =
        if (phoneState != "STANDBY" && phoneState != "OFFLINE") phoneState else faceState(pcState)
}
