package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LanTrustTest {
    private val sha = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    @Test fun onlyPrivateLanIpv4LiteralsCanBePinned() {
        for (host in listOf("192.168.1.20", "10.0.0.7", "172.16.0.1", "172.31.255.254"))
            assertTrue(host, LanTrust.isPrivateLanHost(host))
        for (host in listOf("100.64.0.1", "100.101.2.3", "172.32.0.1", "127.0.0.1", "8.8.8.8",
                "orion-pc.tail1234.ts.net", "192.168.1", "192.168.1.256", "192.168.01.2x", "", "::1",
                "192.168.1.20.evil.example"))
            assertFalse(host, LanTrust.isPrivateLanHost(host))
    }

    @Test fun fingerprintIsTheSha256OfTheCertificateBytes() {
        assertEquals(sha, LanTrust.fingerprint("abc".toByteArray()))
        assertEquals("BA78-16BF-8F01-CFEA", LanTrust.displayCode(sha))
    }

    @Test fun certificateCodeMustBeTypedInFullAndMatch() {
        assertTrue(LanTrust.codeMatches(sha, "BA78-16BF-8F01-CFEA"))
        assertTrue(LanTrust.codeMatches(sha, " ba78 16bf 8f01 cfea "))
        assertTrue(LanTrust.codeMatches(sha, sha))
        assertFalse(LanTrust.codeMatches(sha, "BA78-16BF-8F01"))          // too short to prove anything
        assertFalse(LanTrust.codeMatches(sha, "BA78-16BF-8F01-CFEB"))     // one digit wrong
        assertFalse(LanTrust.codeMatches(sha, ""))
        assertFalse(LanTrust.codeMatches("not-a-hash", "BA78-16BF-8F01-CFEA"))
    }

    @Test fun storedPinsAcceptOnlyLanHostsAndRealHashes() {
        val one = LanTrust.withPin("{}", "192.168.1.20", 8765, sha.uppercase())!!
        assertEquals(mapOf("192.168.1.20:8765" to sha), LanTrust.parsePins(one))
        assertNull(LanTrust.withPin(one, "orion-pc.tail1234.ts.net", 443, sha))
        assertNull(LanTrust.withPin(one, "100.101.2.3", 443, sha))
        assertNull(LanTrust.withPin(one, "192.168.1.21", 8765, "abc"))
        assertNull(LanTrust.withPin(one, "192.168.1.21", 70000, sha))
        val tampered = """{"orion.example:443":"$sha","192.168.1.9:8765":"zz","192.168.1.20:8765":"$sha"}"""
        assertEquals(mapOf("192.168.1.20:8765" to sha), LanTrust.parsePins(tampered))
        assertEquals(emptyMap<String, String>(), LanTrust.parsePins("not json"))
    }
}
