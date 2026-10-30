package com.orion.companion

import java.time.Instant
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PhoneTextModelsTest {
    @Test fun everydayQuestionUsesFastModelFirst() {
        val models = PhoneTextModels.candidates("gemini-3.5-flash-lite")
        assertEquals("gemini-3.5-flash-lite", models.first())
        assertEquals(models.size, models.distinct().size)
    }

    @Test fun fastModelNeverSilentlyUpgradesForLongAnswers() {
        assertEquals(listOf("gemini-3.5-flash-lite", "gemini-3.1-flash-lite"),
            PhoneTextModels.candidates("gemini-3.5-flash-lite"))
    }

    @Test fun chosenModelNeverUsesAnotherPaidModelWithoutConsent() {
        assertEquals(listOf("gemini-2.5-flash"), PhoneTextModels.candidates("gemini-2.5-flash"))
        assertEquals(listOf("gemini-3.1-flash-lite", "gemini-3.5-flash-lite"),
            PhoneTextModels.candidates("gemini-3.1-flash-lite"))
    }

    @Test fun dailyQuotaWaitsForPacificResetRatherThanRetryingInMinutes() {
        val now = Instant.parse("2026-09-29T12:00:00Z").toEpochMilli()
        assertTrue(PhoneTextModels.dailyQuota(
            "GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded"))
        assertFalse(PhoneTextModels.dailyQuota("GenerateRequestsPerMinute limit"))
        val pause = PhoneTextModels.pauseMillis(429, null, true, now)
        assertTrue(pause in 18 * 60 * 60_000L..20 * 60 * 60_000L)
        assertEquals(60_000L, PhoneTextModels.pauseMillis(429, 37L, false, now))
        assertEquals(30_000L, PhoneTextModels.pauseMillis(503, null, false, now))
    }

    @Test fun aPausedModelDoesNotBlockAnAvailableFallback() {
        val now = Instant.parse("2026-09-29T12:00:00Z").toEpochMilli()
        assertEquals(0L, PhoneTextModels.retryDelayMillis(listOf(now + 60_000L, now), now))
        assertEquals(60_000L,
            PhoneTextModels.retryDelayMillis(listOf(now + 60_000L, now + 120_000L), now))
    }
}
