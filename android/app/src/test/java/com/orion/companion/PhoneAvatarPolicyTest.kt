package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Test

class PhoneAvatarPolicyTest {
    @Test fun phoneActivityMapsToTheDesktopStateWords() {
        assertEquals("SPEAKING", PhoneAvatarPolicy.state(true, true, true))
        assertEquals("PROCESSING", PhoneAvatarPolicy.state(true, true, false))
        assertEquals("LISTENING", PhoneAvatarPolicy.state(true, false, false))
        assertEquals("STANDBY", PhoneAvatarPolicy.state(false, false, false))
        assertEquals("OFFLINE", PhoneAvatarPolicy.state(false, false, false, offline = true))
        // Activity wins over being offline: a listening face is still listening.
        assertEquals("LISTENING", PhoneAvatarPolicy.state(true, false, false, offline = true))
    }

    @Test fun stateWordsAreSafeToPlaceInThePageScript() {
        assertEquals("SHUTTING DOWN", PhoneAvatarPolicy.faceState("shutting down"))
        // Quotes, brackets and semicolons never reach the script.
        assertEquals("ALERT", PhoneAvatarPolicy.faceState("');alert(1);//"))
        assertEquals("STANDBY", PhoneAvatarPolicy.faceState("'\\\";()"))
        assertEquals("STANDBY", PhoneAvatarPolicy.faceState(""))
        assertEquals(24, PhoneAvatarPolicy.faceState("X".repeat(100)).length)
    }

    @Test fun linkedFaceFollowsThePcUnlessThePhoneIsBusy() {
        assertEquals("SPEAKING", PhoneAvatarPolicy.linked("SPEAKING", "STANDBY"))
        assertEquals("LISTENING", PhoneAvatarPolicy.linked("SPEAKING", "LISTENING"))
        assertEquals("PROCESSING", PhoneAvatarPolicy.linked("processing", "OFFLINE"))
    }
}
