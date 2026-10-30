package com.orion.companion

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.EOFException
import org.junit.Assert.*
import org.junit.Test

class PcmFramesTest {
    private fun stream(vararg sizes: Int): ByteArray {
        val bytes = ByteArrayOutputStream()
        DataOutputStream(bytes).use { data -> sizes.forEach { size ->
            data.writeInt(size)
            if (size in 1..65_536) data.write(ByteArray(size) { 7 })
        } }
        return bytes.toByteArray()
    }

    @Test fun playsFirstFrameBeforeTheEndMarkerArrives() {
        val frames = PcmFrames(ByteArrayInputStream(stream(4)))
        assertArrayEquals(byteArrayOf(7, 7, 7, 7), frames.next())
        try { frames.next(); fail("Missing end marker was accepted") }
        catch (_: EOFException) { }
    }

    @Test fun recognisesCompleteAudioAndRejectsTruncation() {
        val complete = PcmFrames(ByteArrayInputStream(stream(4, 0)))
        assertEquals(4, complete.next()!!.size)
        assertNull(complete.next())
        val truncated = stream(4).dropLast(1).toByteArray()
        try { PcmFrames(ByteArrayInputStream(truncated)).next(); fail("Truncated PCM was accepted") }
        catch (_: EOFException) { }
    }

    @Test fun refusesInvalidFramesAndUnboundedAudio() {
        for (invalid in listOf(0, -1, 3, 65_538)) {
            try { PcmFrames(ByteArrayInputStream(stream(invalid))).next(); fail("Invalid frame $invalid") }
            catch (_: IllegalArgumentException) { }
            catch (_: IllegalStateException) { }
        }
        val limited = PcmFrames(ByteArrayInputStream(stream(4, 4, 0)), limit = 6)
        limited.next()
        try { limited.next(); fail("Unbounded PCM was accepted") }
        catch (_: IllegalArgumentException) { }
    }
}
