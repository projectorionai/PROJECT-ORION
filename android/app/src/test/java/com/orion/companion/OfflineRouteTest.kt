package com.orion.companion

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayOutputStream
import java.util.Base64
import java.util.zip.GZIPOutputStream

class OfflineRouteTest {
    private val carGpx = """
        <?xml version="1.0" encoding="UTF-8"?>
        <gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1" creator="BRouter">
          <rte>
            <rtept lat="52.0" lon="-1.0"><desc>start</desc><extensions><offset>0</offset></extensions></rtept>
            <rtept lat="52.001" lon="-1.0"><desc>Turn right</desc><extensions><offset>1</offset></extensions></rtept>
            <rtept lat="52.001" lon="-0.999"><desc>destination</desc><extensions><offset>2</offset></extensions></rtept>
          </rte>
          <trk><name>Car route</name><trkseg>
            <trkpt lat="52.0" lon="-1.0"/><trkpt lat="52.001" lon="-1.0"/><trkpt lat="52.001" lon="-0.999"/>
          </trkseg></trk>
        </gpx>
    """.trimIndent()

    private fun fails(action: () -> Unit) {
        try { action(); fail("Expected route rejection") } catch (_: Exception) { }
    }

    @Test fun validBrouterTrackKeepsRealManoeuvreOffsets() {
        val route = OfflineRouteGpx.parse(carGpx)
        assertEquals("Car route", route.name)
        assertEquals(3, route.points.size)
        assertEquals(3, route.manoeuvres.size)
        assertEquals("Turn right", route.manoeuvres[1].instruction)
        assertEquals(1, route.manoeuvres[1].pointIndex)
        assertTrue(route.distanceMetres in 175.0..190.0)
    }

    @Test fun progressUsesRouteDistanceAndUpcomingActualTurn() {
        val route = OfflineRouteGpx.parse(carGpx)
        val progress = route.progress(OfflineCoordinate(52.0005, -1.0))
        assertTrue(progress.distanceFromRouteMetres < 1)
        assertTrue(progress.travelledMetres in 54.0..57.0)
        assertEquals("Turn right", progress.nextManoeuvre!!.instruction)
        assertTrue(progress.nextManoeuvreMetres!! in 54.0..57.0)
        assertEquals(route.distanceMetres, progress.travelledMetres + progress.remainingMetres, 0.001)
    }

    @Test fun offRouteFixIsReportedInsteadOfPretendingItIsOnRoad() {
        val route = OfflineRouteGpx.parse(carGpx)
        assertTrue(route.progress(OfflineCoordinate(52.001, -0.98)).distanceFromRouteMetres > 1_000)
    }

    @Test fun preferredSegmentReacquiresAfterLeavingTheLocalWindow() {
        val points = (0..200).map { OfflineCoordinate(52.0 + it * .001, -1.0) }
        val route = OfflineRoute("Long route", points)
        assertTrue(route.progress(points[190], preferredSegment = 0).segment >= 189)
    }

    @Test fun antimeridianRouteDoesNotAppearToCrossGreenwich() {
        val route = OfflineRoute("Date line", listOf(OfflineCoordinate(0.0, 179.9), OfflineCoordinate(0.0, -179.9)))
        assertTrue(route.progress(OfflineCoordinate(0.0, 180.0)).distanceFromRouteMetres < 1)
        assertTrue(route.progress(OfflineCoordinate(0.0, 0.0)).distanceFromRouteMetres > 19_000_000)
    }

    @Test fun duplicateTrackPointsDoNotDivideByZero() {
        val point = OfflineCoordinate(52.0, -1.0)
        val route = OfflineRoute("Duplicates", listOf(point, point, OfflineCoordinate(52.001, -1.0)))
        val progress = route.progress(point)
        assertTrue(progress.travelledMetres.isFinite())
        assertTrue(progress.distanceFromRouteMetres.isFinite())
    }

    @Test fun disconnectedSegmentsCannotBeDrawnAsAnInventedRoad() {
        fails { OfflineRouteGpx.parse(carGpx.replace("</trkseg>", "</trkseg><trkseg><trkpt lat=\"51.0\" lon=\"0\"/></trkseg>")) }
    }

    @Test fun multipleTracksAreRejected() {
        fails { OfflineRouteGpx.parse(carGpx.replace("</gpx>", "<trk><trkseg><trkpt lat=\"51\" lon=\"0\"/></trkseg></trk></gpx>")) }
    }

    @Test fun outOfRangeOrMismatchedManoeuvreOffsetsAreRejected() {
        fails { OfflineRouteGpx.parse(carGpx.replace("<offset>1</offset>", "<offset>9</offset>")) }
        fails { OfflineRouteGpx.parse(carGpx.replace("<offset>1</offset>", "<offset>0</offset>")) }
    }

    @Test fun waypointOnlyGpxCannotBePresentedAsARoadRoute() {
        fails { OfflineRouteGpx.parse("<gpx><rte><rtept lat=\"52\" lon=\"-1\"><desc>Left</desc></rtept><rtept lat=\"52.001\" lon=\"-1\"/></rte></gpx>") }
    }

    @Test fun namespacesAndRootAreValidated() {
        fails { OfflineRouteGpx.parse(carGpx.replace("http://www.topografix.com/GPX/1/1", "https://example.invalid/not-gpx")) }
        fails { OfflineRouteGpx.parse("<html><trkpt lat=\"52\" lon=\"-1\"/></html>") }
    }

    @Test fun entitiesAndDoctypeAreRejectedBeforeParserResolution() {
        fails { OfflineRouteGpx.parse("<!DOCTYPE gpx [<!ENTITY xx SYSTEM 'file:///private'>]>" + carGpx) }
        fails { OfflineRouteGpx.parse(carGpx.replace("<trk>", "<!ENTITY x 'hi'><trk>")) }
    }

    @Test fun invalidCoordinatesAndShortRoutesAreRejected() {
        listOf("NaN,-1", "Infinity,-1", "86,-1", "52,181", "52", "52,-1,4")
            .forEach { assertNull(OfflineCoordinate.parse(it)) }
        assertEquals(OfflineCoordinate(52.0, -1.0), OfflineCoordinate.parse(" 52, -1 "))
        fails { OfflineRouteGpx.parse(carGpx.replace("lat=\"52.0\"", "lat=\"NaN\"")) }
        fails { OfflineRoute("Empty", listOf(OfflineCoordinate(52.0, -1.0))) }
        fails { OfflineRoute("Stationary", List(2) { OfflineCoordinate(52.0, -1.0) }) }
    }

    @Test fun malformedXmlAndExcessiveNestingFailClosed() {
        fails { OfflineRouteGpx.parse(carGpx.replace("</trk>", "")) }
        fails { OfflineRouteGpx.parse("<gpx>" + "<x>".repeat(33) + "</x>".repeat(33) + "</gpx>") }
    }

    @Test fun pointAndLabelBudgetsAreEnforced() {
        fails { OfflineRouteGpx.parse("<gpx><trk><name>" + "x".repeat(513) + "</name><trkseg><trkpt lat=\"52\" lon=\"-1\"/><trkpt lat=\"52.001\" lon=\"-1\"/></trkseg></trk></gpx>") }
        fails { OfflineRouteGpx.parse("<gpx><trk><trkseg>" + "<trkpt lat=\"52\" lon=\"-1\"/>".repeat(50_001) + "</trkseg></trk></gpx>") }
    }

    @Test fun compressedBrouterServiceReplyIsDecodedAndParsed() {
        val bytes = ByteArrayOutputStream().apply {
            write("z64".toByteArray())
            GZIPOutputStream(this).use { it.write(carGpx.toByteArray()) }
        }.toByteArray()
        val compressed = Base64.getMimeEncoder().encodeToString(bytes)
        assertEquals(carGpx, OfflineRouteGpx.serviceResult(compressed))
        assertEquals(3, OfflineRouteGpx.parse(OfflineRouteGpx.serviceResult(compressed)).points.size)
    }

    @Test fun brouterErrorsAreNotPresentedAsSuccessfulRoutes() {
        fails { OfflineRouteGpx.serviceResult(null) }
        fails { OfflineRouteGpx.serviceResult("segment E0_N50.rd5 not found") }
        fails { OfflineRouteGpx.serviceResult("ejY0invalid") }
        assertEquals(carGpx, OfflineRouteGpx.serviceResult(carGpx))
    }

    @Test fun decompressionBombAndInvalidUtf8AreRejected() {
        val bytes = ByteArrayOutputStream().apply {
            write("z64".toByteArray())
            GZIPOutputStream(this).use { it.write(ByteArray(OfflineRouteGpx.MAX_BYTES + 1) { 'x'.code.toByte() }) }
        }.toByteArray()
        fails { OfflineRouteGpx.serviceResult(Base64.getEncoder().encodeToString(bytes)) }
        fails { OfflineRouteGpx.decodeUtf8(byteArrayOf(0xc3.toByte(), 0x28)) }
    }
}
