package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PrefsAddressTest {
    @Test fun pastedAddressKeepsOnlyAnHttpsOrigin() {
        val parsed = Prefs.parseAddress("https://orion.tail1234.ts.net:443/")
        assertEquals("orion.tail1234.ts.net", parsed?.host)
        assertEquals(443, parsed?.port)
        assertTrue(parsed?.https == true)
        assertEquals("https://orion.tail1234.ts.net/",
            Prefs.canonicalHttpsOrigin("https://orion.tail1234.ts.net:443/"))
    }

    @Test fun cleartextCannotBecomeALearnedEndpoint() {
        assertNull(Prefs.canonicalHttpsOrigin("http://orion.tail1234.ts.net/"))
        assertFalse(Prefs.parseAddress("http://orion.tail1234.ts.net/")?.https ?: true)
        assertNull(Prefs.canonicalHttpsOrigin("https://100.96.1.2/"))
    }

    @Test fun hiddenDestinationsAndPathsAreRejected() {
        val bad = listOf(
            "https://orion.example@attacker.example/",
            "https://orion.example/api/control",
            "https://orion.example/?token=value",
            "https://orion.example/#fragment",
            "https://orion.example/\\evil",
        )
        for (url in bad) {
            assertNull(url, Prefs.canonicalHttpsOrigin(url))
            assertNull(url, Prefs.parseAddress(url))
        }
        assertNull(Prefs.parseAddress("https://[::1]suffix"))
    }
}
