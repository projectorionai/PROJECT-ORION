package com.orion.companion

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

class OfflineModeTest {
    private val now = ZonedDateTime.of(2026, 10, 29, 14, 5, 0, 0, ZoneId.of("Europe/London"))

    @Test fun phoneAnswersTimeDateAndBatteryItself() {
        assertEquals("It's 14:05.", OfflineAnswers.answerFor("What time is it?", now) { 80 })
        assertEquals("It's 14:05.", OfflineAnswers.answerFor("ORION, what's the time", now) { 80 })
        assertEquals("It's Thursday 29 October 2026.", OfflineAnswers.answerFor("what day is it", now) { 80 })
        assertEquals("The battery is at 80%.", OfflineAnswers.answerFor("battery", now) { 80 })
        assertEquals("I can't read the battery right now.", OfflineAnswers.answerFor("how much battery is left", now) { null })
    }

    @Test fun anythingElseWaitsForTheHostedAi() {
        assertNull(OfflineAnswers.answerFor("what time is the meeting tomorrow", now) { 80 })
        assertNull(OfflineAnswers.answerFor("text Mum that I'm late", now) { 80 })
        assertNull(OfflineAnswers.answerFor("", now) { 80 })
    }

    @Test fun queuedRequestsAreBoundedOrderedAndExpire() {
        var clock = 0L
        val queue = OfflineQueue { clock }
        for (i in 1..OfflineQueue.CAPACITY) assertTrue(queue.add("request $i"))
        assertFalse(queue.add("one too many"))                  // refused, nothing evicted
        assertFalse(queue.add("   "))
        assertEquals(OfflineQueue.CAPACITY, queue.size)
        assertEquals("request 1", queue.poll())
        clock += OfflineQueue.MAX_AGE_MS + 1                    // asked too long ago
        assertEquals(0, queue.size)
        assertNull(queue.poll())
        assertTrue(queue.add("fresh"))
        queue.clear()
        assertFalse(queue.isNotEmpty())
    }

    @Test fun cachedFaceStyleAcceptsOnlyKnownDesignsAndPlainColours() {
        val style = FaceStyle.parse(JSONObject("""{"type":"face","design":"Arc","primary":"#FF0000","accent":"red","background":"#000000"}"""))
        assertEquals(FaceStyle("arc", "#ff0000", FaceStyle.DEFAULT.accent, "#000000"), style)
        assertNull(FaceStyle.parse(JSONObject("""{"design":"<script>"}""")))
        assertNull(FaceStyle.parse(null))
    }

    @Test fun lanRediscoveryScansTheOwnSubnetNearestFirst() {
        val hosts = LanDiscovery.subnetHosts(byteArrayOf(192.toByte(), 168.toByte(), 1, 20), 24)
        assertEquals(253, hosts.size)                            // .1-.254 without the phone itself
        assertFalse("192.168.1.20" in hosts)
        assertFalse("192.168.1.0" in hosts || "192.168.1.255" in hosts)
        assertEquals(setOf("192.168.1.19", "192.168.1.21"), hosts.take(2).toSet())
        // A wide network is searched as the phone's own /24, never 65 000 hosts.
        assertEquals(253, LanDiscovery.subnetHosts(byteArrayOf(10, 0, 5, 9), 16).size)
        assertEquals(1021, LanDiscovery.subnetHosts(byteArrayOf(10, 0, 5, 9), 22).size)
    }
}
