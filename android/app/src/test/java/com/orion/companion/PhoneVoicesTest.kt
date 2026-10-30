package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class PhoneVoicesTest {
    @Test fun supportsThirtyDocumentedCloudVoicesAndExplicitPhoneSpeech() {
        assertEquals(30, PhoneVoices.cloud.size)
        assertEquals(30, PhoneVoices.cloud.map { it.name }.toSet().size)
        assertTrue(PhoneVoices.valid("Phone"))
        assertTrue(PhoneVoices.valid("Orus"))
        assertTrue(PhoneVoices.valid("Algieba"))
        assertTrue(PhoneVoices.valid("Schedar"))
        assertFalse(PhoneVoices.valid("unknown"))
        assertFalse(PhoneVoices.valid("Charon\nignored"))
        assertEquals("Charon", PhoneVoices.cloudName("Phone"))
    }
}
