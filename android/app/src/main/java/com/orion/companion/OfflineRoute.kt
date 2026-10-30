package com.orion.companion

import org.xml.sax.Attributes
import org.xml.sax.InputSource
import org.xml.sax.SAXException
import org.xml.sax.SAXParseException
import org.xml.sax.helpers.DefaultHandler
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.Base64
import java.util.zip.GZIPInputStream
import javax.xml.parsers.SAXParserFactory
import kotlin.math.*

/** Coordinates and route maths do not require an Android service or an internet connection. */
internal data class OfflineCoordinate(val latitude: Double, val longitude: Double) {
    init {
        require(latitude.isFinite() && latitude in -85.0511287798..85.0511287798 &&
            longitude.isFinite() && longitude in -180.0..180.0) { "Invalid map coordinate" }
    }

    companion object {
        fun parse(text: String): OfflineCoordinate? {
            val values = text.trim().split(',')
            if (values.size != 2) return null
            val lat = values[0].trim().toDoubleOrNull() ?: return null
            val lon = values[1].trim().toDoubleOrNull() ?: return null
            return runCatching { OfflineCoordinate(lat, lon) }.getOrNull()
        }
    }
}

internal data class OfflineManoeuvre(val pointIndex: Int, val instruction: String)

/** One connected GPX track; disconnected segments and sparse waypoint-only routes are rejected. */
internal class OfflineRoute(
    val name: String,
    val points: List<OfflineCoordinate>,
    val manoeuvres: List<OfflineManoeuvre> = emptyList()
) {
    val cumulativeMetres: DoubleArray
    val distanceMetres: Double get() = cumulativeMetres.last()

    init {
        require(points.size in 2..OfflineRouteGpx.MAX_POINTS) { "A route needs 2–50,000 points" }
        require(manoeuvres.size <= OfflineRouteGpx.MAX_MANOEUVRES)
        require(manoeuvres.zipWithNext().all { it.first.pointIndex <= it.second.pointIndex })
        require(manoeuvres.all { it.pointIndex in points.indices && it.instruction.length in 1..160 })
        cumulativeMetres = DoubleArray(points.size)
        for (index in 1 until points.size) {
            cumulativeMetres[index] = cumulativeMetres[index - 1] + distance(points[index - 1], points[index])
        }
        require(distanceMetres >= 1.0) { "The route is too short" }
    }

    /** Reports along-route distance; it does not guess an ETA, road restrictions or traffic. */
    fun progress(position: OfflineCoordinate, preferredSegment: Int? = null): Progress {
        var best: Projection? = null
        fun consider(index: Int) {
            val projection = project(position, points[index], points[index + 1], index)
            if (best == null || projection.distanceMetres < best!!.distanceMetres) best = projection
        }
        // At a crossing, prefer the local section whilst it remains within the route corridor.
        // This also bounds normal per-fix work on a long route. Reacquisition scans the full track.
        if (preferredSegment != null) {
            for (index in max(0, preferredSegment - 8)..min(points.lastIndex - 1, preferredSegment + 80)) {
                consider(index)
            }
        }
        if (best == null || best!!.distanceMetres > 60.0) {
            best = null
            for (index in 0 until points.lastIndex) consider(index)
        }
        val match = best!!
        val travelled = cumulativeMetres[match.segment] +
            (cumulativeMetres[match.segment + 1] - cumulativeMetres[match.segment]) * match.fraction
        val next = manoeuvres.firstOrNull { cumulativeMetres[it.pointIndex] > travelled + 8.0 }
        return Progress(match.segment, match.distanceMetres, travelled,
            max(0.0, distanceMetres - travelled), next,
            next?.let { max(0.0, cumulativeMetres[it.pointIndex] - travelled) })
    }

    data class Progress(val segment: Int, val distanceFromRouteMetres: Double,
        val travelledMetres: Double, val remainingMetres: Double,
        val nextManoeuvre: OfflineManoeuvre?, val nextManoeuvreMetres: Double?)

    private data class Projection(val segment: Int, val fraction: Double, val distanceMetres: Double)

    companion object {
        private const val RADIUS_METRES = 6_371_008.8

        fun distance(a: OfflineCoordinate, b: OfflineCoordinate): Double {
            val lat1 = Math.toRadians(a.latitude)
            val lat2 = Math.toRadians(b.latitude)
            val dLat = lat2 - lat1
            val dLon = Math.toRadians(b.longitude - a.longitude)
            val h = sin(dLat / 2).pow(2) + cos(lat1) * cos(lat2) * sin(dLon / 2).pow(2)
            return 2 * RADIUS_METRES * asin(sqrt(h.coerceIn(0.0, 1.0)))
        }

        private fun project(p: OfflineCoordinate, a: OfflineCoordinate, b: OfflineCoordinate,
            index: Int): Projection {
            val scale = cos(Math.toRadians(p.latitude))
            fun x(lon: Double): Double {
                var delta = lon - p.longitude
                if (delta > 180) delta -= 360.0
                if (delta < -180) delta += 360.0
                return Math.toRadians(delta) * RADIUS_METRES * scale
            }
            fun y(lat: Double) = Math.toRadians(lat - p.latitude) * RADIUS_METRES
            val ax = x(a.longitude); val ay = y(a.latitude)
            var longitudeStep = b.longitude - a.longitude
            if (longitudeStep > 180) longitudeStep -= 360.0
            if (longitudeStep < -180) longitudeStep += 360.0
            val bx = ax + Math.toRadians(longitudeStep) * RADIUS_METRES * scale
            val by = y(b.latitude)
            val dx = bx - ax; val dy = by - ay
            val squareLength = dx * dx + dy * dy
            val fraction = if (squareLength < 0.001) 0.0
                else (-(ax * dx + ay * dy) / squareLength).coerceIn(0.0, 1.0)
            return Projection(index, fraction, hypot(ax + fraction * dx, ay + fraction * dy))
        }
    }
}

/** Bounded GPX reader; no DTDs, external entities, file references or network resolution. */
internal object OfflineRouteGpx {
    const val MAX_BYTES = 4 * 1024 * 1024
    const val MAX_POINTS = 50_000
    const val MAX_MANOEUVRES = 4_096
    private const val MAX_SERVICE_TEXT = 600_000

    fun serviceResult(raw: String?): String {
        val value = raw ?: throw IOException("BRouter returned no route")
        if (value.length > MAX_SERVICE_TEXT) throw IOException("The route reply is too large")
        if (!value.startsWith("ejY0") && !value.startsWith("z64")) {
            if (!value.trimStart().startsWith('<')) {
                throw IOException("BRouter: ${value.take(220).replace(Regex("[\\r\\n\\t]+"), " ")}")
            }
            if (value.toByteArray(StandardCharsets.UTF_8).size > MAX_BYTES) throw IOException("The route is too large")
            return value
        }
        // BRouter base64-encodes the complete z64 + gzip byte sequence.
        val compressed = try { Base64.getMimeDecoder().decode(value) }
            catch (error: IllegalArgumentException) { throw IOException("Invalid compressed route", error) }
        if (compressed.size < 4 || compressed[0] != 'z'.code.toByte() ||
            compressed[1] != '6'.code.toByte() || compressed[2] != '4'.code.toByte()) {
            throw IOException("Invalid compressed route header")
        }
        return GZIPInputStream(ByteArrayInputStream(compressed, 3, compressed.size - 3)).use { input ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(8192)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                if (output.size() + count > MAX_BYTES) throw IOException("The expanded route is too large")
                output.write(buffer, 0, count)
            }
            decodeUtf8(output.toByteArray())
        }
    }

    fun decodeUtf8(bytes: ByteArray): String {
        require(bytes.size <= MAX_BYTES) { "GPX exceeds 4 MB" }
        return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT).decode(java.nio.ByteBuffer.wrap(bytes)).toString()
    }

    fun parse(xml: String): OfflineRoute {
        require(xml.length <= MAX_BYTES && xml.toByteArray(StandardCharsets.UTF_8).size <= MAX_BYTES) { "GPX exceeds 4 MB" }
        require(!Regex("<!\\s*(DOCTYPE|ENTITY)", RegexOption.IGNORE_CASE).containsMatchIn(xml)) { "GPX must not contain a DTD or entities" }
        val handler = GpxHandler()
        val factory = SAXParserFactory.newInstance().apply { isNamespaceAware = true }
        // Some Android SAX implementations do not expose every optional feature. The entity
        // resolver and declaration rejection remain mandatory independently of these switches.
        listOf("http://xml.org/sax/features/external-general-entities",
            "http://xml.org/sax/features/external-parameter-entities",
            "http://apache.org/xml/features/nonvalidating/load-external-dtd").forEach { feature ->
            runCatching { factory.setFeature(feature, false) }
        }
        val reader = factory.newSAXParser().xmlReader
        reader.entityResolver = org.xml.sax.EntityResolver { _, _ -> throw SAXException("External GPX entities are forbidden") }
        reader.contentHandler = handler
        reader.errorHandler = handler
        reader.parse(InputSource(xml.reader()))
        return handler.route()
    }

    private class GpxHandler : DefaultHandler() {
        private val stack = mutableListOf<String>()
        private val track = mutableListOf<OfflineCoordinate>()
        private val routePoints = mutableListOf<RoutePoint>()
        private var currentRoutePoint: RoutePoint? = null
        private var trackCount = 0
        private var segmentCount = 0
        private var routeCount = 0
        private var routeName = "Saved route"
        private var capture: String? = null
        private var captureDepth = 0
        private val captured = StringBuilder()
        private data class RoutePoint(val coordinate: OfflineCoordinate,
            var description: String = "", var offset: Int? = null)

        override fun startElement(uri: String?, localName: String?, qName: String?, attributes: Attributes) {
            val tag = localName?.takeIf { it.isNotEmpty() } ?: qName.orEmpty()
            val parent = stack.lastOrNull()
            if (stack.isEmpty()) {
                require(tag == "gpx" && uri.orEmpty() in setOf("", "http://www.topografix.com/GPX/1/1", "http://www.topografix.com/GPX/1/0")) { "The file is not GPX" }
            }
            require(stack.size < 32) { "GPX nesting is too deep" }
            stack += tag
            when {
                tag == "trk" && parent == "gpx" -> { trackCount++; require(trackCount == 1) { "Import one connected track at a time" } }
                tag == "trkseg" && parent == "trk" -> { segmentCount++; require(segmentCount == 1) { "Disconnected GPX segments are not supported" } }
                tag == "rte" && parent == "gpx" -> { routeCount++; require(routeCount == 1) { "Import one GPX route at a time" } }
                tag == "trkpt" && parent == "trkseg" -> {
                    require(track.size < MAX_POINTS) { "GPX has more than 50,000 points" }
                    track += coordinate(attributes)
                }
                tag == "rtept" && parent == "rte" -> {
                    require(routePoints.size < MAX_MANOEUVRES) { "GPX has too many route points" }
                    currentRoutePoint = RoutePoint(coordinate(attributes)).also { routePoints += it }
                }
                tag == "name" && parent in setOf("trk", "rte", "metadata") -> beginCapture("name")
                tag == "desc" && parent == "rtept" -> beginCapture("desc")
                tag == "offset" && parent == "extensions" && stack.size >= 3 && stack[stack.size - 3] == "rtept" -> beginCapture("offset")
            }
        }

        private fun coordinate(attributes: Attributes): OfflineCoordinate = OfflineCoordinate(
            attributes.getValue("lat")?.toDoubleOrNull() ?: throw IllegalArgumentException("GPX latitude is missing"),
            attributes.getValue("lon")?.toDoubleOrNull() ?: throw IllegalArgumentException("GPX longitude is missing"))

        private fun beginCapture(field: String) { capture = field; captureDepth = stack.size; captured.setLength(0) }

        override fun characters(ch: CharArray, start: Int, length: Int) {
            if (capture != null) {
                require(captured.length + length <= 512) { "GPX label is too long" }
                captured.append(ch, start, length)
            }
        }

        override fun endElement(uri: String?, localName: String?, qName: String?) {
            if (capture != null && captureDepth == stack.size) {
                val text = captured.toString().trim().replace(Regex("\\s+"), " ")
                when (capture) {
                    "name" -> if (text.isNotEmpty()) routeName = text.take(80)
                    "desc" -> currentRoutePoint?.description = text.take(160)
                    "offset" -> currentRoutePoint?.offset = text.toIntOrNull()
                }
                capture = null
            }
            if (stack.last() == "rtept") currentRoutePoint = null
            stack.removeAt(stack.lastIndex)
        }

        override fun error(error: SAXParseException) { throw error }
        override fun fatalError(error: SAXParseException) { throw error }

        fun route(): OfflineRoute {
            require(trackCount == 1 && segmentCount == 1 && track.size >= 2) {
                "GPX needs one connected track; waypoint-only routes do not contain the road geometry"
            }
            val points = track
            val manoeuvres = routePoints.mapNotNull { point ->
                val index = point.offset ?: return@mapNotNull null
                require(index in track.indices) { "GPX instruction offset is outside the track" }
                require(OfflineRoute.distance(point.coordinate, track[index]) <= 30) { "GPX instruction does not match its track point" }
                point.description.takeIf { it.isNotBlank() }?.let { OfflineManoeuvre(index, it) }
            }
            return OfflineRoute(routeName, points, manoeuvres)
        }
    }
}
