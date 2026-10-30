package com.orion.companion

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Guidance from real route geometry and supplied manoeuvres, without Android or network calls.
 * A service must pass elapsed-realtime values for both clocks and own any route recalculation.
 * This class is confined to one worker; it does not infer turns from an imported track.
 */
internal class OfflineGuidance(private val route: OfflineRoute, private val allowReroute: Boolean) {
    data class Fix(val coordinate: OfflineCoordinate, val accuracyMetres: Double,
        val elapsedRealtimeMillis: Long)

    data class Update(val progress: OfflineRoute.Progress?, val status: String,
        val speech: String?, val requestReroute: Boolean, val arrived: Boolean)

    // One small integer per supplied manoeuvre; old announcements survive GPS loss/backtracking.
    private val announcedStages = IntArray(route.manoeuvres.size) { index ->
        if (route.manoeuvres[index].pointIndex == 0 ||
            route.manoeuvres[index].pointIndex == route.points.lastIndex ||
            route.manoeuvres[index].instruction.isBlank()) PASSED_STAGE else 0
    }
    private var lastTimestamp: Long? = null
    private var lastValidatedFix: Fix? = null
    private var lastUpdate: Update? = null
    private var lastProgress: OfflineRoute.Progress? = null
    private var matchedProgress: OfflineRoute.Progress? = null
    private var matchedFix: Fix? = null
    private var furthestMetres = 0.0
    private var offRouteFirstMillis: Long? = null
    private var offRouteFirstNowMillis: Long? = null
    private var offRouteLastMillis: Long? = null
    private var offRouteFixes = 0
    private var offRouteAnnounced = false
    private var lastOffRouteSpeechMillis: Long? = null
    private var lastRerouteMillis: Long? = null
    private var arrivalAnnounced = false

    fun update(fix: Fix, nowMillis: Long): Update {
        val invalidStatus = when {
            !fix.accuracyMetres.isFinite() || fix.accuracyMetres !in 0.0..MAX_ACCURACY_METRES ->
                "Waiting for an accurate GPS fix"
            nowMillis < 0 || fix.elapsedRealtimeMillis < 0 -> "GPS time is unavailable — guidance paused"
            fix.elapsedRealtimeMillis - nowMillis > MAX_FUTURE_MILLIS ->
                "GPS clock mismatch — guidance paused"
            nowMillis - fix.elapsedRealtimeMillis > MAX_FIX_AGE_MILLIS -> "GPS fix is stale — guidance paused"
            else -> null
        }
        if (invalidStatus != null) {
            resetOffRouteEvidence()
            return suspended(invalidStatus)
        }
        if (lastTimestamp == fix.elapsedRealtimeMillis && lastValidatedFix == fix) {
            // A fresh watchdog repeat does not invalidate a direction already being spoken.
            // Re-emitting its speech or reroute flag would, however, create a feedback loop.
            return lastUpdate?.copy(speech = null, requestReroute = false)
                ?: suspended("Waiting for a newer GPS fix")
        }
        if (lastTimestamp?.let { fix.elapsedRealtimeMillis <= it } == true) {
            return suspended("Waiting for a newer GPS fix")
        }
        lastTimestamp = fix.elapsedRealtimeMillis
        lastValidatedFix = fix
        if (arrivalAnnounced) return remember(Update(lastProgress, "Arrived", null, false, true))

        var progress = route.progress(fix.coordinate, matchedProgress?.segment)
        val corridor = max(30.0, fix.accuracyMetres)
        val previousProgress = matchedProgress
        val previousFix = matchedFix
        if (previousProgress != null && previousFix != null && progress.distanceFromRouteMetres <= corridor) {
            val displacement = OfflineRoute.distance(previousFix.coordinate, fix.coordinate)
            val plausibleAdvance = max(80.0,
                displacement * 2.5 + previousFix.accuracyMetres + fix.accuracyMetres + 20.0)
            if (abs(progress.travelledMetres - previousProgress.travelledMetres) > plausibleAdvance) {
                // Nearby roads may cross much later in the same track. Keep the nearby, continuous
                // route section when it still fits, rather than jumping to a later turn or endpoint.
                val continuous = continuityProgress(fix.coordinate, previousProgress, plausibleAdvance)
                if (continuous != null && continuous.distanceFromRouteMetres <= corridor) {
                    progress = continuous
                } else if (fix.elapsedRealtimeMillis - previousFix.elapsedRealtimeMillis <= MAX_FIX_AGE_MILLIS) {
                    resetOffRouteEvidence()
                    return suspended("Checking the route position")
                }
                // After a real GPS gap, a distant match can be reacquired. Announcement stages
                // are retained, and manoeuvres already passed are retired without late speech.
            }
        }
        lastProgress = progress

        val offRouteThreshold = max(60.0, 2.0 * fix.accuracyMetres)
        if (progress.distanceFromRouteMetres > offRouteThreshold) {
            val previousOffRoute = offRouteLastMillis
            if (previousOffRoute == null || fix.elapsedRealtimeMillis - previousOffRoute > MAX_FIX_AGE_MILLIS) {
                resetOffRouteEvidence()
                offRouteFirstMillis = fix.elapsedRealtimeMillis
                offRouteFirstNowMillis = nowMillis
            }
            offRouteLastMillis = fix.elapsedRealtimeMillis
            offRouteFixes = min(REQUIRED_OFF_ROUTE_FIXES, offRouteFixes + 1)
            val sustained = offRouteFixes >= REQUIRED_OFF_ROUTE_FIXES &&
                fix.elapsedRealtimeMillis - (offRouteFirstMillis ?: fix.elapsedRealtimeMillis) >= OFF_ROUTE_DURATION_MILLIS &&
                nowMillis >= (offRouteFirstNowMillis ?: nowMillis) &&
                nowMillis - (offRouteFirstNowMillis ?: nowMillis) >= OFF_ROUTE_DURATION_MILLIS
            if (!sustained) return remember(Update(progress, "Checking whether you are off route", null, false, false))
            val reroute = allowReroute && cooldownComplete(lastRerouteMillis, nowMillis)
            if (reroute) lastRerouteMillis = nowMillis
            val speech = if (!offRouteAnnounced && cooldownComplete(lastOffRouteSpeechMillis, nowMillis)) {
                offRouteAnnounced = true
                lastOffRouteSpeechMillis = nowMillis
                if (reroute) "Off route. Updating the route." else "You are off the saved route."
            } else null
            val status = when {
                reroute -> "Off route — updating route"
                allowReroute -> "Off route — waiting before another route check"
                else -> "Off the saved route"
            }
            return remember(Update(progress, status, speech, reroute, false))
        }
        resetOffRouteEvidence()
        if (progress.distanceFromRouteMetres > corridor) {
            return remember(Update(progress, "Move closer to the route — guidance paused", null, false, false))
        }
        matchedProgress = progress
        matchedFix = fix

        val backwards = progress.travelledMetres < furthestMetres -
            max(BACKTRACK_TOLERANCE_METRES, fix.accuracyMetres)
        // Retire turns only after the conservative progress bound has passed them. One noisy
        // fix projecting beyond a junction must not silence its final approach instruction.
        furthestMetres = max(furthestMetres, max(0.0, progress.travelledMetres - fix.accuracyMetres))
        for (index in route.manoeuvres.indices) {
            if (route.cumulativeMetres[route.manoeuvres[index].pointIndex] < furthestMetres - TURN_GRACE_METRES) {
                announcedStages[index] = PASSED_STAGE
            }
        }

        // Being geographically near the endpoint is insufficient on a loop or crossing.
        if (!backwards && progress.remainingMetres <= ARRIVAL_ALONG_ROUTE_METRES &&
            progress.distanceFromRouteMetres <= ARRIVAL_RADIUS_METRES &&
            OfflineRoute.distance(fix.coordinate, route.points.last()) <= ARRIVAL_RADIUS_METRES) {
            arrivalAnnounced = true
            return remember(Update(progress, "Arrived", "You have arrived.", false, true))
        }

        val nextIndex = route.manoeuvres.indices.firstOrNull { index ->
            announcedStages[index] < PASSED_STAGE &&
                route.cumulativeMetres[route.manoeuvres[index].pointIndex] >= progress.travelledMetres - TURN_GRACE_METRES
        }
        if (nextIndex == null) {
            return remember(Update(progress, if (backwards) "Rejoin the route in its planned direction" else "Following saved route",
                null, false, false))
        }
        val manoeuvre = route.manoeuvres[nextIndex]
        val distance = max(0.0, route.cumulativeMetres[manoeuvre.pointIndex] - progress.travelledMetres)
        // OfflineRoute.progress intentionally skips turns within 8 m; guidance must still show
        // and announce that supplied instruction once, including a fix just past the turn.
        progress = progress.copy(nextManoeuvre = manoeuvre, nextManoeuvreMetres = distance)
        lastProgress = progress
        val stage = when {
            distance <= 45.0 -> 3
            distance <= 180.0 -> 2
            distance <= 500.0 -> 1
            else -> 0
        }
        val speech = if (!backwards && stage > announcedStages[nextIndex]) {
            announcedStages[nextIndex] = stage
            val instruction = manoeuvre.instruction.trim().replace(Regex("\\s+"), " ")
                .trimEnd('.', '!', '?')
            // The stages control frequency, not the spoken distance. A first fix 70 m away
            // must not claim the turn is 180 m away merely because that stage was crossed.
            if (distance <= TURN_GRACE_METRES) "$instruction."
            else "In ${max(10, (distance / 10.0).roundToInt() * 10)} metres, $instruction."
        } else null
        return remember(Update(progress, if (backwards) "Rejoin the route in its planned direction" else "Following route",
            speech, false, false))
    }

    private fun remember(update: Update): Update { lastUpdate = update; return update }

    private fun suspended(status: String) = remember(Update(null, status, null, false, false))

    private fun resetOffRouteEvidence() {
        offRouteFirstMillis = null
        offRouteFirstNowMillis = null
        offRouteLastMillis = null
        offRouteFixes = 0
        offRouteAnnounced = false
    }

    private fun cooldownComplete(previousMillis: Long?, nowMillis: Long): Boolean =
        previousMillis == null || (nowMillis >= previousMillis && nowMillis - previousMillis >= REROUTE_COOLDOWN_MILLIS)

    /** Bounded projection near the last segment, additionally constrained by along-track continuity. */
    private fun continuityProgress(position: OfflineCoordinate, previous: OfflineRoute.Progress,
        advance: Double): OfflineRoute.Progress? {
        var best: OfflineRoute.Progress? = null
        val scale = cos(Math.toRadians(position.latitude))
        for (index in max(0, previous.segment - 8)..min(route.points.lastIndex - 1, previous.segment + 80)) {
            val a = route.points[index]
            val b = route.points[index + 1]
            fun longitudeDelta(value: Double): Double {
                var delta = value
                if (delta > 180.0) delta -= 360.0
                if (delta < -180.0) delta += 360.0
                return delta
            }
            val ax = Math.toRadians(longitudeDelta(a.longitude - position.longitude)) * RADIUS_METRES * scale
            val ay = Math.toRadians(a.latitude - position.latitude) * RADIUS_METRES
            val dx = Math.toRadians(longitudeDelta(b.longitude - a.longitude)) * RADIUS_METRES * scale
            val dy = Math.toRadians(b.latitude - a.latitude) * RADIUS_METRES
            val squareLength = dx * dx + dy * dy
            val fraction = if (squareLength < .001) 0.0 else (-(ax * dx + ay * dy) / squareLength).coerceIn(0.0, 1.0)
            val travelled = route.cumulativeMetres[index] +
                (route.cumulativeMetres[index + 1] - route.cumulativeMetres[index]) * fraction
            if (abs(travelled - previous.travelledMetres) > advance) continue
            val distance = hypot(ax + fraction * dx, ay + fraction * dy)
            if (best == null || distance < best.distanceFromRouteMetres) {
                best = OfflineRoute.Progress(index, distance, travelled,
                    max(0.0, route.distanceMetres - travelled), null, null)
            }
        }
        return best
    }

    companion object {
        private const val MAX_ACCURACY_METRES = 50.0
        private const val MAX_FIX_AGE_MILLIS = 15_000L
        private const val MAX_FUTURE_MILLIS = 2_000L
        private const val REQUIRED_OFF_ROUTE_FIXES = 3
        private const val OFF_ROUTE_DURATION_MILLIS = 8_000L
        private const val REROUTE_COOLDOWN_MILLIS = 60_000L
        private const val TURN_GRACE_METRES = 8.0
        private const val BACKTRACK_TOLERANCE_METRES = 8.0
        private const val ARRIVAL_RADIUS_METRES = 20.0
        private const val ARRIVAL_ALONG_ROUTE_METRES = 30.0
        private const val PASSED_STAGE = 4
        private const val RADIUS_METRES = 6_371_008.8
    }
}
