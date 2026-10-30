package com.orion.companion

import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneVoiceCooldownTest {
    @Test fun dailyVoiceQuotaResumesAtTheExistingPacificReset() {
        val now = Instant.parse("2026-09-29T12:00:00Z").toEpochMilli()
        val delay = PhoneVoiceCooldown.delayMillis(429,
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded", null, now)
        assertTrue(delay in 18 * 60 * 60_000L..20 * 60 * 60_000L)
    }

    @Test fun minuteVoiceQuotaHonoursAReportedWait() {
        val now = Instant.parse("2026-09-29T12:00:00Z").toEpochMilli()
        assertEquals(3 * 60_000L,
            PhoneVoiceCooldown.delayMillis(429, "GenerateRequestsPerMinute limit", 10, now))
        assertEquals(7 * 60_000L,
            PhoneVoiceCooldown.delayMillis(429, "GenerateRequestsPerMinute limit", 420, now))
    }
}
