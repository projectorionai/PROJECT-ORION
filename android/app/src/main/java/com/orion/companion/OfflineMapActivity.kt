package com.orion.companion

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.EditText
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import org.mapsforge.core.graphics.Style
import org.mapsforge.core.model.LatLong
import org.mapsforge.map.android.graphics.AndroidGraphicFactory
import org.mapsforge.map.android.graphics.AndroidBitmap
import org.mapsforge.map.android.util.AndroidUtil
import org.mapsforge.map.android.view.MapView
import org.mapsforge.map.datastore.MapDataStore
import org.mapsforge.map.datastore.MultiMapDataStore
import org.mapsforge.map.layer.overlay.Circle
import org.mapsforge.map.layer.overlay.Marker
import org.mapsforge.map.layer.overlay.Polyline
import org.mapsforge.map.layer.renderer.TileRendererLayer
import org.mapsforge.map.reader.MapFile
import org.mapsforge.map.rendertheme.internal.MapsforgeThemes
import org.mapsforge.core.util.MercatorProjection
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import java.util.Locale
import kotlin.math.roundToInt

/** Phone-local maps, GPS, saved destinations and routes. BRouter supplies the offline car graph. */
class OfflineMapActivity : AppCompatActivity(), LocationListener {
    private lateinit var mapView: MapView
    private lateinit var packs: OfflineMapPacks
    private lateinit var locationManager: LocationManager
    private lateinit var mapStatus: TextView
    private lateinit var positionStatus: TextView
    private lateinit var transferStatus: TextView
    private lateinit var transferProgress: ProgressBar
    private lateinit var cancelTransfer: Button
    private lateinit var followButton: Button
    private lateinit var routeStatus: TextView
    private lateinit var router: OfflineCarRouter
    private lateinit var places: OfflinePlaces
    private lateinit var routeStore: OfflineRouteStore
    private val navigationWorker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "orion-offline-navigation").apply { isDaemon = true }
    }
    private var route: OfflineRoute? = null
    private var displayedRouteRevision: Long? = null
    private var guidanceRequestedAfterRoute = false
    private var resumed = false
    private var lastGuidanceStatus: String? = null
    private var guidanceStoppedStatus: String? = null
    private var routeLine: Polyline? = null
    private var preferredSegment: Int? = null
    @Volatile private var routeGeneration = 0L
    private var progressGeneration = 0L
    private var routeSource = "Saved GPX"
    private var routeCalculating = false
    @Volatile private var activityClosed = false
    private var pendingCoordinateRoute = false
    private var dataStore: MapDataStore? = null
    private var marker: Marker? = null
    private var accuracyCircle: Circle? = null
    private var lastFix: Location? = null
    private var lastRenderedBearingBucket: Int? = null
    private var follow = true
    private var locationPermissionAsked = false
    private var requestedCoordinate: LatLong? = null
    private var pendingImport: OfflineMapPacks.Pack? = null
    private val main = Handler(Looper.getMainLooper())

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { grants ->
        if (grants.values.any { it }) startLocation()
        else positionStatus.text = "Location permission denied. The offline map is still available."
    }
    private val importDocument = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        val pack = pendingImport
        pendingImport = null
        if (uri != null && pack != null) beginImport(pack, uri)
    }
    private val importRoute = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) readRouteDocument(uri)
    }
    private val ageTicker = object : Runnable {
        override fun run() {
            if (isDestroyed || !resumed) return
            updateGuidanceSnapshot()
            updatePositionStatus()
            lastFix?.let(::updateRouteProgress)
            main.postDelayed(this, 1_000)
        }
    }
    private val guidanceNotifications = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        startGuidanceWithPermissions()
    }
    private val guidanceLocation = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED) startGuidanceWithPermissions()
        else routeStatus.text = "Precise Location permission is needed for guidance · Allow it in Android settings"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        AndroidGraphicFactory.createInstance(application)
        packs = OfflineMapPacks(this)
        router = OfflineCarRouter(this)
        places = OfflinePlaces(File(filesDir, "offline-navigation/places.bin"))
        routeStore = OfflineRouteStore(File(filesDir, "offline-navigation/route.gpx"))
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager
        locationPermissionAsked = savedInstanceState?.getBoolean("location_permission_asked") ?: false
        requestedCoordinate = parseCoordinate(intent.getStringExtra("coordinate_query"))
        pendingCoordinateRoute = requestedCoordinate != null && intent.getBooleanExtra("route_to_coordinate", false)
        follow = requestedCoordinate == null
        buildScreen()
        loadMap()
        restoreSavedRoute()
        if (pendingCoordinateRoute) routeStatus.text = "Waiting for a fresh GPS fix before routing…"
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        main.removeCallbacks(ageTicker)
        ageTicker.run()
        startLocation()
    }

    override fun onPause() {
        resumed = false
        main.removeCallbacks(ageTicker)
        locationManager.removeUpdates(this)
        super.onPause()
    }

    override fun onDestroy() {
        activityClosed = true
        routeGeneration++
        progressGeneration++
        main.removeCallbacks(ageTicker)
        router.close()
        navigationWorker.shutdownNow()
        packs.close()
        mapView.destroyAll()
        dataStore?.close()
        AndroidGraphicFactory.clearResourceMemoryCache()
        super.onDestroy()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        outState.putBoolean("location_permission_asked", locationPermissionAsked)
        super.onSaveInstanceState(outState)
    }

    private fun buildScreen() {
        val root = FrameLayout(this).apply { setBackgroundColor(Color.rgb(13, 19, 25)) }
        mapView = MapView(this).apply {
            setBuiltInZoomControls(true)
            getMapScaleBar().setVisible(true)
            setCenter(LatLong(54.5, -3.5))
            setZoomLevel(6.toByte())
        }
        root.addView(mapView, FrameLayout.LayoutParams(-1, -1))

        val top = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(9), dp(12), dp(8))
            setBackgroundColor(Color.rgb(15, 19, 24))
        }
        val buttons = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        buttons.addView(button("Back") { finish() })
        buttons.addView(label("ORION MAP", 18f).apply {
            setTextColor(Color.WHITE)
            setTypeface(null, android.graphics.Typeface.BOLD)
        }, LinearLayout.LayoutParams(0, -2, 1f))
        buttons.addView(button("Packs") { showPackMenu() })
        top.addView(buttons)
        mapStatus = label("Offline vector road map", 12f).apply { setTextColor(Color.rgb(174, 174, 183)) }
        top.addView(mapStatus)
        root.addView(top, FrameLayout.LayoutParams(-1, -2, Gravity.TOP))

        val bottom = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(12))
            setBackgroundColor(Color.rgb(15, 19, 24))
        }
        positionStatus = label("Waiting for phone location permission", 13f)
        bottom.addView(positionStatus)
        val mapActions = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        followButton = button(if (follow) "Follow on" else "Follow off") { toggleFollow() }
        mapActions.addView(followButton)
        mapActions.addView(button("Places") { showPlacesMenu() })
        mapActions.addView(button("Routes", accent = true) { showRoutesMenu() })
        bottom.addView(mapActions)
        routeStatus = label("No route selected", 12f)
        bottom.addView(routeStatus)
        transferStatus = label("", 12f).apply { visibility = View.GONE }
        bottom.addView(transferStatus)
        transferProgress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            visibility = View.GONE
            isIndeterminate = true
        }
        bottom.addView(transferProgress, LinearLayout.LayoutParams(-1, dp(5)))
        cancelTransfer = button("Cancel transfer") { packs.cancel() }.apply { visibility = View.GONE }
        bottom.addView(cancelTransfer)
        bottom.addView(label("Map data © OpenStreetMap contributors", 11f).apply {
            setTextColor(Color.rgb(174, 174, 183))
        })
        root.addView(bottom, FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
    }

    private fun loadMap() {
        val files = packs.installedFiles()
        val unreadable = OfflineMapPacks.Pack.entries.filter { pack ->
            packs.file(pack).exists() && !files.contains(packs.file(pack))
        }
        if (files.isEmpty()) {
            mapStatus.text = if (unreadable.isEmpty()) "No map pack · Tap Packs to add one"
                else "Map pack unreadable · Replace it in Packs"
            centreRequestedCoordinate()
            return
        }
        val store = MultiMapDataStore(MultiMapDataStore.DataPolicy.DEDUPLICATE)
        val usable = mutableListOf<String>()
        for (file in files) {
            runCatching {
                val source = MapFile(file)
                store.addMapDataStore(source, false, false)
                usable += file.nameWithoutExtension
            }.onFailure { error ->
                mapStatus.text = "Cannot open ${file.name}: ${error.message ?: "invalid map"}. Reimport it in Map packs."
            }
        }
        if (usable.isEmpty()) { store.close(); centreRequestedCoordinate(); return }
        dataStore = store
        val cache = AndroidUtil.createTileCache(this, "orion-offline-vector",
            mapView.model.displayModel.tileSize, 1f,
            mapView.model.frameBufferModel.overdrawFactor)
        val layer = TileRendererLayer(cache, store, mapView.model.mapViewPosition, AndroidGraphicFactory.INSTANCE)
        layer.setXmlRenderTheme(MapsforgeThemes.DEFAULT)
        mapView.layerManager.layers.add(layer)
        addPositionLayers()
        mapStatus.text = if (unreadable.isNotEmpty()) "Road map loaded · One pack unreadable"
            else if (usable.size == 2) "Offline road map · UK coverage"
            else "Offline road map · ${if (usable[0] == "great-britain") "Great Britain" else "Northern Ireland"}"
        if (!centreRequestedCoordinate()) {
            val start = lastFix?.takeIf { !isStale(it) && store.boundingBox().contains(it.latitude, it.longitude) }
            if (start != null) {
                mapView.setCenter(LatLong(start.latitude, start.longitude))
                mapView.setZoomLevel(14.toByte())
            } else {
                mapView.setCenter(store.boundingBox().centerPoint)
                mapView.setZoomLevel(6.toByte())
            }
        }
    }

    private fun centreRequestedCoordinate(): Boolean {
        val point = requestedCoordinate ?: return false
        mapView.setCenter(point)
        mapView.setZoomLevel(13.toByte())
        if (dataStore?.boundingBox()?.contains(point) == false) {
            mapStatus.text = "Requested point is outside the saved map"
        }
        positionStatus.text = "Requested point: ${format(point.latitude)}, ${format(point.longitude)} · not GPS"
        return true
    }

    private fun parseCoordinate(raw: String?): LatLong? {
        val values = raw?.trim()?.split(',') ?: return null
        if (values.size != 2) return null
        val lat = values[0].trim().toDoubleOrNull() ?: return null
        val lon = values[1].trim().toDoubleOrNull() ?: return null
        if (!lat.isFinite() || !lon.isFinite() ||
            lat !in MercatorProjection.LATITUDE_MIN..MercatorProjection.LATITUDE_MAX ||
            lon !in -180.0..180.0) return null
        return LatLong(lat, lon)
    }

    private fun addPositionLayers() {
        val factory = AndroidGraphicFactory.INSTANCE
        val fill = factory.createPaint().apply {
            setColor(Color.argb(48, 42, 137, 226))
            setStyle(Style.FILL)
        }
        val outline = factory.createPaint().apply {
            setColor(Color.argb(190, 32, 113, 211))
            setStyle(Style.STROKE)
            setStrokeWidth(dp(2).toFloat())
        }
        accuracyCircle = Circle(null, 0f, fill, outline).also { mapView.layerManager.layers.add(it) }
        marker = Marker(null, makePositionBitmap(null), 0, 0).also { mapView.layerManager.layers.add(it) }
        lastFix?.let(::updateMarker)
    }

    private fun makePositionBitmap(bearing: Float?): org.mapsforge.core.graphics.Bitmap {
        val side = dp(42)
        val bitmap = Bitmap.createBitmap(side, side, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val centre = side / 2f
        val white = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; style = Paint.Style.FILL }
        val blue = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(16, 100, 213); style = Paint.Style.FILL }
        val shadow = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.argb(70, 0, 0, 0); style = Paint.Style.FILL }
        canvas.drawCircle(centre, centre + dp(1), dp(15).toFloat(), shadow)
        canvas.drawCircle(centre, centre, dp(15).toFloat(), white)
        canvas.drawCircle(centre, centre, dp(12).toFloat(), blue)
        if (bearing != null) {
            canvas.save()
            canvas.rotate(bearing, centre, centre)
            val arrow = Path().apply {
                moveTo(centre, centre - dp(9))
                lineTo(centre + dp(6), centre + dp(7))
                lineTo(centre, centre + dp(4))
                lineTo(centre - dp(6), centre + dp(7))
                close()
            }
            canvas.drawPath(arrow, white)
            canvas.restore()
        } else {
            canvas.drawCircle(centre, centre, dp(5).toFloat(), white)
        }
        return AndroidBitmap(bitmap)
    }

    private fun startLocation() {
        val fine = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
        val coarse = ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED
        if (!fine && !coarse) {
            positionStatus.text = "Location permission needed to show the moving phone position."
            if (!locationPermissionAsked) {
                locationPermissionAsked = true
                permissions.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
            }
            return
        }
        locationManager.removeUpdates(this)
        var started = false
        try {
            if (fine && locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1_000L, 1f, this, Looper.getMainLooper())
                locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)?.let(::onLocationChanged)
                started = true
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 2_000L, 5f, this, Looper.getMainLooper())
                locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)?.let(::onLocationChanged)
                started = true
            }
        } catch (_: SecurityException) {
            positionStatus.text = "Location permission is unavailable."
            return
        }
        if (!started) positionStatus.text = "Turn on phone location to show live position. The map itself works offline."
    }

    override fun onLocationChanged(location: Location) {
        if (runCatching { OfflineCoordinate(location.latitude, location.longitude) }.isFailure ||
            location.elapsedRealtimeNanos <= 0 ||
            location.elapsedRealtimeNanos > SystemClock.elapsedRealtimeNanos() + 2_000_000_000L ||
            (location.hasAccuracy() && (!location.accuracy.isFinite() || location.accuracy <= 0))) return
        val previous = lastFix
        if (previous != null && location.elapsedRealtimeNanos < previous.elapsedRealtimeNanos) return
        if (previous != null && !isStale(previous) && previous.hasAccuracy() && location.hasAccuracy() &&
            location.accuracy > previous.accuracy * 3f &&
            location.elapsedRealtimeNanos - previous.elapsedRealtimeNanos < 10_000_000_000L) return
        lastFix = location
        updateMarker(location)
        updatePositionStatus()
        updateRouteProgress(location)
        if (pendingCoordinateRoute && !isStale(location) && location.hasAccuracy() && location.accuracy <= 100) {
            pendingCoordinateRoute = false
            guidanceRequestedAfterRoute = true
            requestedCoordinate?.let { calculateRoute(OfflineCoordinate(it.latitude, it.longitude)) }
        }
        if (follow && !isStale(location) &&
            dataStore?.boundingBox()?.contains(location.latitude, location.longitude) == true) {
            mapView.setCenter(LatLong(location.latitude, location.longitude))
            if (mapView.model.mapViewPosition.zoomLevel < 13) mapView.setZoomLevel(15.toByte())
        }
    }

    private fun updateMarker(location: Location) {
        if (dataStore?.boundingBox()?.contains(location.latitude, location.longitude) != true) {
            marker?.setLatLong(null)
            accuracyCircle?.setLatLong(null)
            mapView.layerManager.redrawLayers()
            return
        }
        val point = LatLong(location.latitude, location.longitude)
        marker?.setLatLong(point)
        val bearingBucket = if (location.hasBearing() && location.bearing.isFinite()) (location.bearing / 5f).roundToInt() else -1
        if (lastRenderedBearingBucket != bearingBucket) {
            marker?.setBitmap(makePositionBitmap(location.bearing.takeIf { location.hasBearing() && it.isFinite() }))
            lastRenderedBearingBucket = bearingBucket
        }
        accuracyCircle?.setLatLong(point)
        accuracyCircle?.setRadius(if (location.hasAccuracy()) location.accuracy.coerceIn(0f, 10_000f) else 0f)
        mapView.layerManager.redrawLayers()
    }

    private fun updatePositionStatus() {
        val fix = lastFix ?: return
        val ageSeconds = ((SystemClock.elapsedRealtimeNanos() - fix.elapsedRealtimeNanos) / 1_000_000_000L).coerceAtLeast(0)
        val parts = mutableListOf("Phone ${format(fix.latitude)}, ${format(fix.longitude)}")
        if (fix.hasAccuracy()) parts += "±${fix.accuracy.roundToInt()} m"
        if (fix.hasSpeed() && fix.speed.isFinite() && fix.speed >= 0) parts += "${"%.0f".format(Locale.UK, fix.speed * 3.6f)} km/h"
        if (fix.hasBearing() && fix.bearing.isFinite()) parts += "${fix.bearing.roundToInt()}° heading"
        if (ageSeconds > STALE_SECONDS) parts += "STALE (${ageSeconds}s old)"
        else parts += "live"
        if (dataStore != null && dataStore?.boundingBox()?.contains(fix.latitude, fix.longitude) == false) {
            parts += "outside saved map"
        }
        positionStatus.text = parts.joinToString("  •  ")
    }

    private fun isStale(location: Location): Boolean =
        SystemClock.elapsedRealtimeNanos() - location.elapsedRealtimeNanos > STALE_SECONDS * 1_000_000_000L

    private fun toggleFollow() {
        follow = !follow
        followButton.text = if (follow) "Follow on" else "Follow off"
        if (follow && ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            locationPermissionAsked = false
            startLocation()
        }
        if (follow) lastFix?.takeIf { !isStale(it) && dataStore?.boundingBox()?.contains(it.latitude, it.longitude) == true }?.let {
            mapView.setCenter(LatLong(it.latitude, it.longitude))
            if (mapView.model.mapViewPosition.zoomLevel < 13) mapView.setZoomLevel(15.toByte())
        }
    }

    private fun showPackMenu() {
        val choices = OfflineMapPacks.Pack.entries.map { pack ->
            val state = when {
                packs.installed(pack) -> "installed (${sizeLabel(packs.file(pack).length())})"
                packs.file(pack).exists() -> "unreadable — replace"
                else -> "not installed"
            }
            "${pack.label}: $state"
        }.toTypedArray()
        AlertDialog.Builder(this).setTitle("Offline vector map packs")
            .setItems(choices) { _, index -> showPackActions(OfflineMapPacks.Pack.entries[index]) }
            .setNeutralButton("About") { _, _ ->
                AlertDialog.Builder(this).setTitle("About offline maps")
                    .setMessage("These are road and place maps, not satellite photos. Routes → BRouter setup explains the separate offline road regions needed to calculate car routes in ORION. You can also import a GPX route. Download large maps on unmetered Wi-Fi and keep ORION open during transfer.")
                    .setPositiveButton("Close", null).show()
            }
            .setNegativeButton("Close", null).show()
    }

    private fun showPackActions(pack: OfflineMapPacks.Pack) {
        val installed = packs.installed(pack)
        val choices = if (installed) arrayOf("Update on Wi-Fi", "Import replacement .map", "Remove pack")
            else arrayOf("Download on Wi-Fi", "Import .map from phone")
        AlertDialog.Builder(this).setTitle(pack.label).setItems(choices) { _, index ->
            when {
                index == 0 -> AlertDialog.Builder(this).setTitle("Download ${pack.label}?")
                    .setMessage("Approximately ${pack.approximateSize}. Uses unmetered Wi-Fi and phone storage. Keep ORION open until the download finishes. The existing map remains usable during an update.")
                    .setPositiveButton("Download") { _, _ -> beginDownload(pack) }
                    .setNegativeButton("Cancel", null).show()
                index == 1 -> { pendingImport = pack; importDocument.launch(arrayOf("*/*")) }
                installed && index == 2 -> AlertDialog.Builder(this).setTitle("Remove ${pack.label}?")
                    .setMessage("The map will no longer be available offline until reinstalled.")
                    .setPositiveButton("Remove") { _, _ -> packs.remove(pack) { result ->
                        if (result.isSuccess) recreate() else showTransferError(result.exceptionOrNull())
                    } }
                    .setNegativeButton("Keep", null).show()
            }
        }.setNegativeButton("Close", null).show()
    }

    private fun beginDownload(pack: OfflineMapPacks.Pack) {
        showTransfer("Downloading ${pack.label}")
        packs.download(pack, ::updateTransfer) { finishTransfer(it) }
    }

    private fun beginImport(pack: OfflineMapPacks.Pack, uri: android.net.Uri) {
        showTransfer("Importing ${pack.label}")
        packs.import(pack, uri, ::updateTransfer) { finishTransfer(it) }
    }

    private fun showTransfer(message: String) {
        transferStatus.text = message
        transferStatus.visibility = View.VISIBLE
        transferProgress.visibility = View.VISIBLE
        transferProgress.isIndeterminate = true
        cancelTransfer.visibility = View.VISIBLE
    }

    private fun updateTransfer(progress: OfflineMapPacks.Progress) {
        if (isDestroyed) return
        val total = progress.totalBytes
        transferStatus.text = "${progress.phase} ${progress.pack.label}: ${sizeLabel(progress.bytes)}" +
            (if (total != null) " / ${sizeLabel(total)}" else "")
        transferProgress.isIndeterminate = total == null || total <= 0
        if (total != null && total > 0) {
            transferProgress.max = 1000
            transferProgress.progress = ((progress.bytes.toDouble() / total) * 1000).toInt().coerceIn(0, 1000)
        }
    }

    private fun finishTransfer(result: Result<File>) {
        if (isDestroyed) return
        cancelTransfer.visibility = View.GONE
        transferProgress.visibility = View.GONE
        if (result.isSuccess) recreate() else showTransferError(result.exceptionOrNull())
    }

    private fun showTransferError(error: Throwable?) {
        transferStatus.visibility = View.VISIBLE
        transferStatus.text = error?.message ?: "Map operation failed"
        cancelTransfer.visibility = View.GONE
        transferProgress.visibility = View.GONE
    }

    private fun openOsmAnd() {
        val appIntent = sequenceOf("net.osmand", "net.osmand.plus")
            .mapNotNull { packageManager.getLaunchIntentForPackage(it) }.firstOrNull()
        if (appIntent != null) {
            startActivity(appIntent)
            return
        }
        AlertDialog.Builder(this).setTitle("Car routes use OsmAnd")
            .setMessage("Install OsmAnd and download the regions of your journey for spoken directions. ORION can display a saved GPX route or calculate a car route when BRouter and its road regions are installed.")
            .setPositiveButton("Open app listing") { _, _ ->
                val listing = Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=net.osmand"))
                runCatching { startActivity(listing) }.onFailure {
                    startActivity(Intent(Intent.ACTION_VIEW,
                        Uri.parse("https://play.google.com/store/apps/details?id=net.osmand")))
                }
            }.setNegativeButton("Close", null).show()
    }

    private fun showRoutesMenu() {
        val choices = mutableListOf("Car route to coordinates", "Plan between coordinates", "Import GPX route", "BRouter setup", "Spoken guidance · OsmAnd")
        if (OfflineNavigationService.snapshot.running) choices.add(0, "Stop guidance")
        else if (route != null) choices.add(0, "Start guidance")
        if (route != null) choices += "Clear saved route"
        if (routeCalculating) choices += "Cancel route calculation"
        AlertDialog.Builder(this).setTitle("Offline routes").setItems(choices.toTypedArray()) { _, index ->
            when (choices[index]) {
                "Start guidance" -> startGuidance()
                "Stop guidance" -> OfflineNavigationService.stop(this)
                "Car route to coordinates" -> showDestinationEditor()
                "Plan between coordinates" -> showRoutePlanner()
                "Import GPX route" -> importRoute.launch(arrayOf("application/gpx+xml", "application/xml", "text/xml", "application/octet-stream"))
                "BRouter setup" -> showRouterSetup()
                "Spoken guidance · OsmAnd" -> openOsmAnd()
                "Clear saved route" -> clearRoute()
                "Cancel route calculation" -> router.cancel()
            }
        }.setNegativeButton("Close", null).show()
    }

    private fun showRouterSetup() {
        AlertDialog.Builder(this).setTitle("Offline car routing")
            .setMessage("Install BRouter, open it once and download the road regions covering the whole journey. Ensure its car-fast profile is installed. These .rd5 routing regions are separate from ORION's .map files.\n\nAfter calculating a route, choose Start guidance for offline spoken manoeuvres and GPS tracking when locked. Download an offline UK phone voice in Android's Text-to-speech settings. OsmAnd remains available for its address search and guidance.\n\nSet up the journey whilst parked. Routing data may be out of date; follow road signs and restrictions.")
            .setPositiveButton(if (router.isInstalled()) "Open BRouter" else "Install BRouter") { _, _ ->
                val launch = packageManager.getLaunchIntentForPackage(OfflineCarRouter.PACKAGE)
                if (launch != null) runCatching { startActivity(launch) }.onFailure { navigationError(it) }
                else openListing(OfflineCarRouter.PACKAGE)
            }.setNegativeButton("Close", null).show()
    }

    private fun openListing(packageName: String) {
        runCatching { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("market://details?id=$packageName"))) }
            .onFailure {
                runCatching { startActivity(Intent(Intent.ACTION_VIEW,
                    Uri.parse("https://play.google.com/store/apps/details?id=$packageName"))) }
                    .onFailure { navigationError(it) }
            }
    }

    private fun showPlacesMenu() {
        navigationWork({ places.list() }) { result ->
            val saved = result.getOrElse { navigationError(it); return@navigationWork }
            val items = arrayOf("Save current position", "Add coordinates") + saved.map { it.name }.toTypedArray()
            AlertDialog.Builder(this).setTitle("Saved places").setItems(items) { _, index ->
                when (index) {
                    0 -> {
                        val fix = freshPosition() ?: return@setItems
                        showDestinationEditor(OfflineCoordinate(fix.latitude, fix.longitude))
                    }
                    1 -> showDestinationEditor()
                    else -> showPlaceActions(saved[index - 2])
                }
            }.setNegativeButton("Close", null).show()
        }
    }

    private fun showPlaceActions(place: OfflinePlace) {
        AlertDialog.Builder(this).setTitle(place.name)
            .setItems(arrayOf("Calculate car route", "Show on map", "Remove saved place")) { _, index ->
                when (index) {
                    0 -> calculateRoute(place.coordinate)
                    1 -> {
                        requestedCoordinate = LatLong(place.coordinate.latitude, place.coordinate.longitude)
                        follow = false
                        followButton.text = "Follow off"
                        centreRequestedCoordinate()
                    }
                    2 -> AlertDialog.Builder(this).setTitle("Remove ${place.name}?")
                        .setPositiveButton("Remove") { _, _ ->
                            navigationWork({ places.remove(place.id) }) { it.exceptionOrNull()?.let(::navigationError) }
                        }.setNegativeButton("Keep", null).show()
                }
            }.setNegativeButton("Close", null).show()
    }

    private fun showDestinationEditor(point: OfflineCoordinate? = null) {
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
        }
        val coordinates = EditText(this).apply {
            hint = "Latitude, longitude"
            inputType = android.text.InputType.TYPE_CLASS_TEXT
            isSingleLine = true
            val initial = point ?: requestedCoordinate?.let { OfflineCoordinate(it.latitude, it.longitude) }
            if (initial != null) setText("${format(initial.latitude)}, ${format(initial.longitude)}")
        }
        val name = EditText(this).apply { hint = "Place name (for Save)"; isSingleLine = true }
        panel.addView(coordinates)
        panel.addView(name)
        val dialog = AlertDialog.Builder(this).setTitle("Destination coordinates").setView(panel)
            .setPositiveButton("Route", null).setNeutralButton("Save", null).setNegativeButton("Cancel", null).create()
        dialog.setOnShowListener {
            fun destination(): OfflineCoordinate? = OfflineCoordinate.parse(coordinates.text.toString()).also {
                if (it == null) coordinates.error = "Enter a valid latitude, longitude"
            }
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val target = destination() ?: return@setOnClickListener
                dialog.dismiss(); calculateRoute(target)
            }
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                val target = destination() ?: return@setOnClickListener
                val label = name.text.toString().trim()
                if (label.isBlank() || label.length > 80 || label.any { it.code < 32 }) {
                    name.error = "Enter a place name up to 80 characters"; return@setOnClickListener
                }
                navigationWork({ places.save(label, target) }) { result ->
                    result.onSuccess { routeStatus.text = "Saved ${it.name}"; dialog.dismiss() }
                        .onFailure { navigationError(it) }
                }
            }
        }
        dialog.show()
    }

    private fun freshPosition(): Location? {
        val fix = lastFix?.takeIf { !isStale(it) && it.hasAccuracy() && it.accuracy <= 100 }
        if (fix == null) {
            navigationError(IOException("Wait for a fresh location with accuracy within 100 metres. Turn on precise Location permission and GPS."))
        }
        return fix
    }

    /** A planned start is route input, never substituted for the measured phone position. */
    private fun showRoutePlanner() {
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(8), dp(18), 0)
        }
        fun coordinateField(label: String) = EditText(this).apply {
            hint = label
            inputType = android.text.InputType.TYPE_CLASS_TEXT
            isSingleLine = true
            panel.addView(this)
        }
        val start = coordinateField("Start latitude, longitude")
        val destination = coordinateField("Destination latitude, longitude")
        val dialog = AlertDialog.Builder(this).setTitle("Plan an offline car route")
            .setMessage("Enter both points. This plans the journey without a GPS fix; it does not change your phone's position.")
            .setView(panel).setPositiveButton("Plan", null).setNegativeButton("Cancel", null).create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val origin = OfflineCoordinate.parse(start.text.toString())
                val target = OfflineCoordinate.parse(destination.text.toString())
                if (origin == null) start.error = "Enter a valid latitude, longitude"
                if (target == null) destination.error = "Enter a valid latitude, longitude"
                if (origin != null && target != null) {
                    dialog.dismiss()
                    calculateRoute(origin, target, "BRouter planned route")
                }
            }
        }
        dialog.show()
    }

    private fun calculateRoute(destination: OfflineCoordinate) {
        if (!router.isInstalled()) { showRouterSetup(); return }
        val fix = freshPosition() ?: return
        calculateRoute(OfflineCoordinate(fix.latitude, fix.longitude), destination, "BRouter car route")
    }

    private fun calculateRoute(start: OfflineCoordinate, destination: OfflineCoordinate, source: String) {
        if (!router.isInstalled()) { showRouterSetup(); return }
        if (routeCalculating) { navigationError(IOException("A route calculation is already running")); return }
        val generation = ++routeGeneration
        OfflineNavigationService.stop(this)
        guidanceStoppedStatus = null
        val storageRevision = routeStore.beginChange()
        routeCalculating = true
        routeStatus.text = "Calculating offline car route…"
        router.calculate(start, destination) { result ->
            routeCalculating = false
            if (activityClosed || generation != routeGeneration) return@calculate
            result.onSuccess { xml -> persistRoute(xml, source, generation, storageRevision) }
                .onFailure { navigationError(it) }
        }
    }

    private fun readRouteDocument(uri: Uri) {
        OfflineNavigationService.stop(this)
        guidanceRequestedAfterRoute = false
        guidanceStoppedStatus = null
        if (routeCalculating) { router.cancel(); routeCalculating = false }
        val generation = ++routeGeneration
        val storageRevision = routeStore.beginChange()
        routeStatus.text = "Reading GPX route…"
        navigationWork({
            val input = contentResolver.openInputStream(uri) ?: throw IOException("The selected GPX cannot be read")
            val bytes = input.use { source ->
                val output = java.io.ByteArrayOutputStream()
                val buffer = ByteArray(8192)
                while (true) {
                    if (Thread.currentThread().isInterrupted) throw IOException("Route import cancelled")
                    val count = source.read(buffer)
                    if (count < 0) break
                    if (output.size() + count > OfflineRouteGpx.MAX_BYTES) throw IOException("GPX exceeds 4 MB")
                    output.write(buffer, 0, count)
                }
                output.toByteArray()
            }
            OfflineRouteGpx.decodeUtf8(bytes)
        }) { result ->
            if (generation != routeGeneration) return@navigationWork
            result.onSuccess { persistRoute(it, "Imported GPX", generation, storageRevision) }.onFailure { navigationError(it) }
        }
    }

    private fun restoreSavedRoute() {
        val generation = routeGeneration
        navigationWork({ routeStore.readSnapshot() }) { result ->
            if (generation != routeGeneration) return@navigationWork
            result.onSuccess { (restored, revision) ->
                if (routeStore.isCurrent(revision)) restored?.let { showRoute(it, "Saved GPX", revision) }
            }
                .onFailure { navigationError(it) }
        }
    }

    private fun persistRoute(xml: String, source: String, generation: Long, storageRevision: Long) {
        navigationWork({
            routeStore.save(xml, storageRevision) { !activityClosed && generation == routeGeneration }
        }) { result ->
            if (generation != routeGeneration) return@navigationWork
            result.onSuccess {
                if (!routeStore.isCurrent(storageRevision)) return@onSuccess
                showRoute(it, source, storageRevision)
                if (guidanceRequestedAfterRoute) {
                    guidanceRequestedAfterRoute = false
                    if (resumed) startGuidance()
                    else routeStatus.text = "Route ready · Open Routes to start guidance"
                }
            }.onFailure { navigationError(it) }
        }
    }

    private fun showRoute(value: OfflineRoute, source: String, revision: Long) {
        route = value
        displayedRouteRevision = revision
        preferredSegment = null
        routeSource = source
        follow = true
        followButton.text = "Follow on"
        routeLine?.let { mapView.layerManager.layers.remove(it) }
        val paint = AndroidGraphicFactory.INSTANCE.createPaint().apply {
            setColor(Color.rgb(170, 31, 49)); setStyle(Style.STROKE); setStrokeWidth(dp(5).toFloat())
        }
        routeLine = Polyline(paint, AndroidGraphicFactory.INSTANCE).apply {
            setPoints(value.points.map { LatLong(it.latitude, it.longitude) })
        }.also { line ->
            // Keep the position marker and its accuracy disc above the route.
            val layers = mapView.layerManager.layers
            layers.add((layers.size() - if (marker != null) 2 else 0).coerceAtLeast(0), line)
        }
        routeStatus.text = "$source · ${distanceLabel(value.distanceMetres)} · ${value.manoeuvres.size} instructions"
        if (dataStore == null) mapStatus.text = "Route loaded · Install map packs to see the roads"
        val start = value.points.first()
        mapView.setCenter(LatLong(start.latitude, start.longitude))
        mapView.setZoomLevel(14.toByte())
        mapView.layerManager.redrawLayers()
        lastFix?.let(::updateRouteProgress)
    }

    private fun updateRouteProgress(fix: Location) {
        if (OfflineNavigationService.snapshot.running) {
            updateGuidanceSnapshot()
            return
        }
        if (guidanceStoppedStatus != null) { routeStatus.text = guidanceStoppedStatus; return }
        val current = route ?: return
        val generation = ++progressGeneration
        if (isStale(fix) || !fix.hasAccuracy() || fix.accuracy > 100) {
            routeStatus.text = "$routeSource · Waiting for an accurate, fresh GPS fix"; return
        }
        val preferred = preferredSegment
        navigationWork({ current.progress(OfflineCoordinate(fix.latitude, fix.longitude), preferred) }) { result ->
            if (generation != progressGeneration || route !== current) return@navigationWork
            result.onSuccess { progress ->
                preferredSegment = progress.segment
                val corridor = maxOf(60.0, fix.accuracy.toDouble() * 2)
                routeStatus.text = when {
                    progress.distanceFromRouteMetres > corridor ->
                        "Off route by ${distanceLabel(progress.distanceFromRouteMetres)} · Recalculate in Routes"
                    progress.remainingMetres < 25 && OfflineRoute.distance(
                        OfflineCoordinate(fix.latitude, fix.longitude), current.points.last()) < 50 -> "Destination nearby"
                    progress.nextManoeuvre != null ->
                        "${distanceLabel(progress.nextManoeuvreMetres!!)} · ${progress.nextManoeuvre.instruction} · ${distanceLabel(progress.remainingMetres)} remaining"
                    else -> "${distanceLabel(progress.remainingMetres)} remaining · Follow saved route"
                }
            }.onFailure { navigationError(it) }
        }
    }

    private fun clearRoute() {
        OfflineNavigationService.stop(this)
        guidanceRequestedAfterRoute = false
        guidanceStoppedStatus = null
        router.cancel(); routeCalculating = false
        routeGeneration++; progressGeneration++
        val storageRevision = routeStore.beginChange()
        route = null; preferredSegment = null; displayedRouteRevision = null
        routeLine?.let { mapView.layerManager.layers.remove(it) }
        routeLine = null
        mapView.layerManager.redrawLayers()
        routeStatus.text = "No route selected"
        navigationWork({ routeStore.clear(storageRevision) }) {
            it.exceptionOrNull()?.let(::navigationError)
        }
    }

    private fun startGuidance() {
        if (!resumed || route == null || routeCalculating) return
        if (android.os.Build.VERSION.SDK_INT >= 33 && ContextCompat.checkSelfPermission(this,
                Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            guidanceNotifications.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else startGuidanceWithPermissions()
    }

    private fun startGuidanceWithPermissions() {
        if (!resumed) return
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) !=
            PackageManager.PERMISSION_GRANTED) {
            routeStatus.text = "Allow precise Location permission, then choose Start guidance."
            guidanceLocation.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION))
            return
        }
        val revision = displayedRouteRevision
        if (revision == null || !routeStore.isCurrent(revision)) {
            routeStatus.text = "The saved route changed · Select the current journey before starting guidance"
            restoreSavedRoute()
            return
        }
        OfflineNavigationService.start(this, routeSource.startsWith("BRouter"), revision)
            .onSuccess { guidanceStoppedStatus = null; lastGuidanceStatus = "Starting navigation" }
            .onFailure { navigationError(it) }
    }

    private fun updateGuidanceSnapshot() {
        val snapshot = OfflineNavigationService.snapshot
        if (!snapshot.running) {
            if (lastGuidanceStatus != null && snapshot.status != lastGuidanceStatus) {
                guidanceStoppedStatus = snapshot.status
                routeStatus.text = snapshot.status
                lastGuidanceStatus = snapshot.status
            }
            return
        }
        lastGuidanceStatus = snapshot.status
        guidanceStoppedStatus = null
        val revision = snapshot.routeRevision
        val current = snapshot.route
        if (revision != null && current != null && revision != displayedRouteRevision && routeStore.isCurrent(revision)) {
            showRoute(current, if (snapshot.allowReroute) "BRouter car route" else "Saved GPX", revision)
        }
        val progress = snapshot.progress
        val description = if (progress != null && snapshot.status.startsWith("Following")) {
            val next = progress.nextManoeuvre
            if (next != null && progress.nextManoeuvreMetres != null)
                "${distanceLabel(progress.nextManoeuvreMetres)} · ${next.instruction} · ${distanceLabel(progress.remainingMetres)} remaining"
            else "${distanceLabel(progress.remainingMetres)} remaining"
        } else snapshot.status
        routeStatus.text = if (!snapshot.speechAvailable && snapshot.speechStatus != null)
            "$description · ${snapshot.speechStatus}" else description
    }

    private fun <T> navigationWork(work: () -> T, done: (Result<T>) -> Unit) {
        if (activityClosed) return
        navigationWorker.execute {
            val result = runCatching(work)
            main.post { if (!activityClosed && !isDestroyed) done(result) }
        }
    }

    private fun navigationError(error: Throwable) {
        routeStatus.text = error.message?.take(240) ?: "Offline navigation operation failed"
    }

    private fun distanceLabel(metres: Double): String = if (metres < 1_000)
        "${(metres / 10).roundToInt() * 10} m" else String.format(Locale.UK, "%.1f km", metres / 1_000)

    private fun label(value: String, size: Float) = TextView(this).apply {
        text = value
        textSize = size
        setTextColor(Color.WHITE)
        setPadding(dp(4), dp(5), dp(4), dp(5))
    }

    private fun button(value: String, accent: Boolean = false, action: () -> Unit) = Button(this).apply {
        text = value
        isAllCaps = false
        setTextColor(Color.WHITE)
        backgroundTintList = android.content.res.ColorStateList.valueOf(
            if (accent) Color.rgb(164, 32, 42) else Color.rgb(42, 42, 48))
        setOnClickListener { action() }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).roundToInt()
    private fun format(value: Double) = String.format(Locale.UK, "%.5f", value)
    private fun sizeLabel(bytes: Long) = if (bytes >= 1_000_000_000L)
        String.format(Locale.UK, "%.2f GB", bytes / 1_000_000_000.0)
    else String.format(Locale.UK, "%.0f MB", bytes / 1_000_000.0)

    companion object { private const val STALE_SECONDS = 30L }
}
