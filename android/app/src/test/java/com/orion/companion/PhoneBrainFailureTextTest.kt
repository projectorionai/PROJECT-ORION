package com.orion.companion

import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneBrainFailureTextTest {
    @Test fun networkFailureExplainsWhichConnectionNeedsAttention() {
        assertTrue(PhoneBrain.failureText(UnknownHostException("private-hostname"))
            .contains("mobile data or Wi-Fi"))
        assertTrue(PhoneBrain.failureText(SocketTimeoutException("private-hostname"))
            .contains("timed out"))
    }

    @Test fun unexpectedExceptionDetailsAreNeverDisplayed() {
        val diagnostic = PhoneBrain.failureText(IOException("credential=secret-value"))
        assertFalse(diagnostic.contains("secret-value"))
        assertTrue(diagnostic.contains("IOException"))
    }
}
