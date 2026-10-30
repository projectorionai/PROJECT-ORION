package com.orion.companion

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.cos

class OfflineGuidanceTest {
    private fun point(northMetres: Double, eastMetres: Double = 0.0) = OfflineCoordinate(
        52.0 + northMetres / 111_195.0802335329,
        -1.0 + eastMetres / (111_195.0802335329 * cos(Math.toRadians(52.0))))

    private fun route(withTurn: Boolean = true) = OfflineRoute("Test route",
        (0..100).map { point(it * 10.0) },
        if (withTurn) listOf(OfflineManoeuvre(60, "Turn left")) else emptyList())

    private fun update(engine: OfflineGuidance, north: Double, time: Long,
        east: Double = 0.0, accuracy: Double = 5.0, now: Long = time) =
        engine.update(OfflineGuidance.Fix(point(north, east), accuracy, time), now)

    @Test fun approachStagesSpeakOnceAndKeepOnlyTheNearestReachedStage() {
        val engine = OfflineGuidance(route(), false)
        assertNull(update(engine, 50.0, 1_000).speech)
        assertTrue(update(engine, 150.0, 2_000).speech!!.contains("Turn left"))
        assertNull(update(engine, 160.0, 3_000).speech)
        assertTrue(update(engine, 470.0, 4_000).speech!!.contains("Turn left"))
        assertNull(update(engine, 480.0, 5_000).speech)
        assertTrue(update(engine, 570.0, 6_000).speech!!.contains("Turn left"))
        assertNull(update(engine, 580.0, 7_000).speech)

        val nearStart = OfflineGuidance(route(), false)
        assertTrue(update(nearStart, 570.0, 1_000).speech!!.contains("Turn left"))
        assertNull(update(nearStart, 575.0, 2_000).speech)
        assertNull(update(nearStart, 490.0, 3_000).speech)
    }

    @Test fun turnWithinEightMetresIsNotLostByTheRouteProgressFilter() {
        val track = route()
        assertNull(track.progress(point(595.0)).nextManoeuvre)
        val result = update(OfflineGuidance(track, false), 595.0, 1_000)
        assertEquals("Turn left.", result.speech)
        assertEquals("Turn left", result.progress!!.nextManoeuvre!!.instruction)
        assertTrue(result.progress.nextManoeuvreMetres!! < 8.0)
    }

    @Test fun firstFixInsideAnApproachStageUsesItsActualDistanceNotTheStageBoundary() {
        val result = update(OfflineGuidance(route(), false), 530.0, 1_000)
        assertEquals("In 70 metres, Turn left.", result.speech)
    }

    @Test fun aFixJustPastTheTurnCanStillSpeakButALateFixCannot() {
        assertEquals("Turn left.", update(OfflineGuidance(route(), false), 604.0, 1_000).speech)
        assertNull(update(OfflineGuidance(route(), false), 610.0, 1_000).speech)
    }

    @Test fun backtrackingAndGpsReacquisitionDoNotRepeatAnOldTurn() {
        val engine = OfflineGuidance(route(), false)
        assertNotNull(update(engine, 595.0, 1_000).speech)
        assertNull(update(engine, 585.0, 2_000).speech)
        assertNull(update(engine, 615.0, 3_000).speech)
        assertNull(update(engine, 580.0, 4_000).speech)
        assertNull(update(engine, 570.0, 25_000).speech)
        assertNull(update(engine, 599.0, 26_000).speech)
    }

    @Test fun anUncertainOvershootDoesNotRetireTheFinalApproachInstruction() {
        val engine = OfflineGuidance(route(), false)
        assertNotNull(update(engine, 470.0, 1_000, accuracy = 50.0).speech)
        assertNull(update(engine, 640.0, 2_000, accuracy = 50.0).speech)
        val recovered = update(engine, 580.0, 3_000, accuracy = 50.0)
        assertTrue(recovered.speech!!.contains("Turn left"))
        assertNull(update(engine, 580.0, 4_000, accuracy = 50.0).speech)
    }

    @Test fun closelySpacedActualTurnsEachKeepTheirFinalInstruction() {
        val track = route(false)
        val turns = OfflineRoute(track.name, track.points, listOf(
            OfflineManoeuvre(60, "Turn left"), OfflineManoeuvre(62, "Turn right")))
        val engine = OfflineGuidance(turns, false)
        assertEquals("Turn left.", update(engine, 595.0, 1_000).speech)
        assertTrue(update(engine, 615.0, 2_000).speech!!.contains("Turn right"))
        assertNull(update(engine, 618.0, 3_000).speech)
    }

    @Test fun repeatedAndReversedTimestampsCannotSpeakOrRequestAnotherRoute() {
        val engine = OfflineGuidance(route(), true)
        assertNotNull(update(engine, 150.0, 5_000).speech)
        val duplicate = update(engine, 570.0, 5_000, now = 6_000)
        val older = update(engine, 595.0, 4_000, now = 6_000)
        for (result in listOf(duplicate, older)) {
            assertNull(result.speech)
            assertFalse(result.requestReroute)
            assertFalse(result.arrived)
            assertEquals("Waiting for a newer GPS fix", result.status)
        }
        assertNotNull(update(engine, 570.0, 7_000).speech)
    }

    @Test fun aFreshWatchdogRepeatKeepsTheValidStatusWithoutRepeatingSpeech() {
        val engine = OfflineGuidance(route(), false)
        val original = update(engine, 595.0, 5_000)
        assertNotNull(original.speech)
        val repeated = update(engine, 595.0, 5_000, now = 7_000)
        assertEquals(original.progress, repeated.progress)
        assertEquals(original.status, repeated.status)
        assertNull(repeated.speech)
        assertFalse(repeated.requestReroute)
    }

    @Test fun aStaleWatchdogRepeatInvalidatesPreviouslyValidDirections() {
        val engine = OfflineGuidance(route(), false)
        assertNotNull(update(engine, 595.0, 5_000).progress)
        val stale = update(engine, 595.0, 5_000, now = 20_001)
        assertNull(stale.progress)
        assertNull(stale.speech)
        assertFalse(stale.requestReroute)
        assertEquals("GPS fix is stale — guidance paused", stale.status)
        assertNotNull(update(engine, 596.0, 21_000).progress)
        assertNull(update(engine, 596.0, 22_000).speech)
    }

    @Test fun aPoorFixSuspendsExistingDirectionsUntilANewValidatedFix() {
        val engine = OfflineGuidance(route(), false)
        update(engine, 595.0, 5_000)
        assertNull(update(engine, 595.0, 6_000, accuracy = 100.0).progress)
        assertNull(update(engine, 595.0, 5_000, now = 7_000).progress)
        assertNotNull(update(engine, 596.0, 8_000).progress)
    }

    @Test fun invalidAccuracySuspendsAllInstructionsAndArrival() {
        for (accuracy in listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -1.0, 50.01)) {
            val result = update(OfflineGuidance(route(), true), 1_000.0, 1_000, accuracy = accuracy)
            assertNull(result.speech)
            assertFalse(result.requestReroute)
            assertFalse(result.arrived)
            assertEquals("Waiting for an accurate GPS fix", result.status)
        }
        assertTrue(update(OfflineGuidance(route(), false), 1_000.0, 1_000, accuracy = 50.0).arrived)
        assertTrue(update(OfflineGuidance(route(), false), 1_000.0, 1_000, accuracy = 0.0).arrived)
    }

    @Test fun staleFutureAndNegativeClockValuesSuspendGuidance() {
        val invalid = listOf(
            update(OfflineGuidance(route(), true), 595.0, 1_000, now = 16_001),
            update(OfflineGuidance(route(), true), 595.0, 3_001, now = 1_000),
            update(OfflineGuidance(route(), true), 595.0, -1, now = 1_000),
            update(OfflineGuidance(route(), true), 595.0, 1_000, now = -1))
        for (result in invalid) {
            assertNull(result.speech)
            assertFalse(result.requestReroute)
            assertFalse(result.arrived)
            assertTrue(result.status.contains("paused"))
        }
        assertNotNull(update(OfflineGuidance(route(), false), 595.0, 16_000, now = 31_000).speech)
        assertNotNull(update(OfflineGuidance(route(), false), 595.0, 3_000, now = 1_000).speech)
    }

    @Test fun oneJumpAwayFromTheRouteCannotTriggerAReroute() {
        val engine = OfflineGuidance(route(), true)
        val result = update(engine, 200.0, 1_000, east = 90.0)
        assertNull(result.speech)
        assertFalse(result.requestReroute)
        val sameFixLater = update(engine, 200.0, 1_000, east = 90.0, now = 10_000)
        assertNull(sameFixLater.speech)
        assertFalse(sameFixLater.requestReroute)
    }

    @Test fun sustainedOffRouteEvidenceRequiresThreeDistinctFixesAndEightSeconds() {
        val engine = OfflineGuidance(route(), true)
        assertFalse(update(engine, 200.0, 1_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 201.0, 5_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 202.0, 8_999, east = 90.0).requestReroute)
        val result = update(engine, 203.0, 9_000, east = 90.0)
        assertTrue(result.requestReroute)
        assertEquals("Off route. Updating the route.", result.speech)
        assertFalse(result.arrived)
    }

    @Test fun twoFixesEvenOverEightSecondsAreInsufficient() {
        val engine = OfflineGuidance(route(), true)
        assertFalse(update(engine, 200.0, 1_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 201.0, 9_000, east = 90.0).requestReroute)
        assertTrue(update(engine, 202.0, 10_000, east = 90.0).requestReroute)
    }

    @Test fun permittedFutureTimestampCannotShortenTheEightSecondEvidenceWindow() {
        val engine = OfflineGuidance(route(), true)
        update(engine, 200.0, 1_000, east = 90.0)
        update(engine, 200.0, 5_000, east = 90.0, now = 4_000)
        assertFalse(update(engine, 200.0, 9_000, east = 90.0, now = 7_000).requestReroute)
        assertTrue(update(engine, 200.0, 10_000, east = 90.0, now = 9_000).requestReroute)
    }

    @Test fun routeJitterInsideAccuracyBasedCorridorNeverRequestsAReroute() {
        val engine = OfflineGuidance(route(), true)
        for (time in listOf(1_000L, 5_000L, 9_000L, 13_000L)) {
            val result = update(engine, 200.0, time, east = 80.0, accuracy = 50.0)
            assertNull(result.speech)
            assertFalse(result.requestReroute)
        }
        assertFalse(update(engine, 200.0, 17_000, east = 120.0, accuracy = 50.0).requestReroute)
        assertFalse(update(engine, 200.0, 21_000, east = 120.0, accuracy = 50.0).requestReroute)
        assertTrue(update(engine, 200.0, 25_000, east = 120.0, accuracy = 50.0).requestReroute)
    }

    @Test fun aGoodFixResetsAnOffRouteEpisode() {
        val engine = OfflineGuidance(route(), true)
        update(engine, 200.0, 1_000, east = 90.0)
        update(engine, 200.0, 5_000, east = 90.0)
        update(engine, 200.0, 9_000)
        assertFalse(update(engine, 200.0, 13_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 200.0, 17_000, east = 90.0).requestReroute)
        assertTrue(update(engine, 200.0, 21_000, east = 90.0).requestReroute)
    }

    @Test fun staleOrInvalidFixCannotBridgeAnOffRouteWindow() {
        for (invalidAccuracy in listOf(false, true)) {
            val engine = OfflineGuidance(route(), true)
            update(engine, 200.0, 1_000, east = 90.0)
            update(engine, 200.0, 5_000, east = 90.0)
            val invalid = if (invalidAccuracy) update(engine, 200.0, 9_000, east = 90.0, accuracy = 100.0)
                else update(engine, 200.0, 5_000, east = 90.0, now = 21_000)
            assertFalse(invalid.requestReroute)
            assertFalse(update(engine, 200.0, 25_000, east = 90.0).requestReroute)
            assertFalse(update(engine, 200.0, 30_000, east = 90.0).requestReroute)
            assertTrue(update(engine, 200.0, 33_000, east = 90.0).requestReroute)
        }
    }

    @Test fun absenceOfFixesCannotBridgeAnOffRouteWindow() {
        val engine = OfflineGuidance(route(), true)
        update(engine, 200.0, 1_000, east = 90.0)
        update(engine, 200.0, 5_000, east = 90.0)
        assertFalse(update(engine, 200.0, 30_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 200.0, 34_000, east = 90.0).requestReroute)
        assertTrue(update(engine, 200.0, 38_000, east = 90.0).requestReroute)
    }

    @Test fun rerouteCooldownRequiresSixtySecondsAndANewFix() {
        val engine = OfflineGuidance(route(), true)
        update(engine, 200.0, 1_000, east = 90.0)
        update(engine, 200.0, 5_000, east = 90.0)
        assertTrue(update(engine, 200.0, 9_000, east = 90.0).requestReroute)
        for (time in 13_000L..65_000L step 4_000L) {
            val result = update(engine, 200.0, time, east = 90.0)
            assertFalse(result.requestReroute)
            assertNull(result.speech)
        }
        assertTrue(update(engine, 200.0, 69_000, east = 90.0).requestReroute)
        assertFalse(update(engine, 200.0, 69_000, east = 90.0, now = 70_000).requestReroute)
    }

    @Test fun importedTrackHasNoInventedTurnsOrAutomaticReroute() {
        val engine = OfflineGuidance(route(withTurn = false), false)
        assertNull(update(engine, 200.0, 1_000).speech)
        update(engine, 200.0, 5_000, east = 90.0)
        update(engine, 200.0, 9_000, east = 90.0)
        val result = update(engine, 200.0, 13_000, east = 90.0)
        assertFalse(result.requestReroute)
        assertEquals("You are off the saved route.", result.speech)
        assertNull(result.progress!!.nextManoeuvre)
    }

    @Test fun arrivalNeedsBothTheEndpointAndTheRouteAndSpeaksOnce() {
        val engine = OfflineGuidance(route(), true)
        assertFalse(update(engine, 1_000.0, 1_000, east = 70.0).arrived)
        assertFalse(update(engine, 1_000.0, 2_000, east = 25.0).arrived)
        val arrived = update(engine, 995.0, 3_000)
        assertTrue(arrived.arrived)
        assertEquals("You have arrived.", arrived.speech)
        val repeated = update(engine, 998.0, 4_000)
        assertTrue(repeated.arrived)
        assertNull(repeated.speech)
        assertFalse(repeated.requestReroute)
    }

    @Test fun crossingNearTheEndpointDoesNotJumpAlongTheRouteOrClaimArrival() {
        val loop = OfflineRoute("Crossing", listOf(point(0.0), point(500.0),
            point(500.0, 100.0), point(5.0, 100.0), point(5.0, -100.0), point(5.0)),
            listOf(OfflineManoeuvre(3, "Turn left")))
        val engine = OfflineGuidance(loop, false)
        update(engine, 0.0, 1_000)
        // The later crossing is closer than the first vertical segment, but is a kilometre ahead.
        val raw = loop.progress(point(5.0, .5), preferredSegment = 0)
        assertTrue(raw.travelledMetres > 1_000.0)
        val result = update(engine, 5.0, 2_000, east = .5)
        assertTrue(result.progress!!.travelledMetres < 20.0)
        assertFalse(result.arrived)
        assertNull(result.speech)
    }

    @Test fun initialFixAtALoopStartIsNotArrivalJustBecauseTheEndpointIsNearby() {
        val loop = OfflineRoute("Loop", listOf(point(0.0), point(500.0),
            point(500.0, 100.0), point(0.0)))
        val result = update(OfflineGuidance(loop, false), 0.0, 1_000)
        assertFalse(result.arrived)
        assertNull(result.speech)
        assertTrue(result.progress!!.remainingMetres > 900.0)
    }

    @Test fun routeStartAndDestinationLabelsAreNotSpokenAsTurns() {
        val track = route(false)
        val labelled = OfflineRoute(track.name, track.points, listOf(
            OfflineManoeuvre(0, "start"), OfflineManoeuvre(60, "Turn left"),
            OfflineManoeuvre(100, "destination")))
        val engine = OfflineGuidance(labelled, false)
        assertNull(update(engine, 0.0, 1_000).speech)
        assertEquals("Turn left.", update(engine, 595.0, 2_000).speech)
        assertEquals("You have arrived.", update(engine, 995.0, 3_000).speech)
    }
}
