package com.orion.companion

import android.Manifest
import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.view.View
import androidx.core.content.ContextCompat
import java.io.File
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** User-started GPS guidance with phone-private routes and embedded UK speech only. */
class OfflineNavigationService : Service() {
    internal data class Snapshot(
        val running: Boolean = false,
        val status: String = "Navigation is off",
        val route: OfflineRoute? = null,
        val routeRevision: Long? = null,
        val generation: Long = 0L,
        val coordinate: OfflineCoordinate? = null,
        val progress: OfflineRoute.Progress? = null,
        val lastFixElapsedRealtimeMillis: Long? = null,
        val speechAvailable: Boolean = false,
        val speechStatus: String? = null,
        val allowReroute: Boolean = false
    )

    companion object {
        private const val START = "com.orion.companion.OFFLINE_NAVIGATION_START"
        private const val STOP = "com.orion.companion.OFFLINE_NAVIGATION_STOP"
        private const val CHANNEL = "orion_offline_navigation"
        private const val NOTIFICATION_ID = 410
        private const val EXTRA_REROUTE = "allow_reroute"
        private const val EXTRA_REVISION = "route_revision"
        private const val REROUTE_COOLDOWN_MILLIS = 60_000L
        private const val SPEECH_MAX_WAIT_MILLIS = 8_000L
        private val generations = AtomicLong()
        @Volatile private var instance: OfflineNavigationService? = null
        @Volatile internal var snapshot = Snapshot()
            private set
        val running: Boolean get() = snapshot.running
        val status: String get() = snapshot.status

        /** An explicit tap in a visible Activity starts a location foreground service. */
        fun start(context: Context, allowReroute: Boolean, expectedRevision: Long): Result<Unit> = runCatching {
            val activity = context as? Activity
                ?: throw IllegalStateException("Open the map before starting navigation")
            check(!activity.isFinishing && !activity.isDestroyed &&
                activity.window.decorView.isShown && activity.window.decorView.windowVisibility == View.VISIBLE &&
                activity.hasWindowFocus()) {
                "Open the map before starting navigation"
            }
            check(ContextCompat.checkSelfPermission(activity, Manifest.permission.ACCESS_FINE_LOCATION) ==
                PackageManager.PERMISSION_GRANTED) { "Allow precise location before starting navigation" }
            require(expectedRevision >= 0L) { "Select a saved route before starting navigation" }
            val store = OfflineRouteStore(File(activity.filesDir, "offline-navigation/route.gpx"))
            check(store.isCurrent(expectedRevision)) { "The route changed; select the latest saved route" }
            ContextCompat.startForegroundService(activity,
                Intent(activity, OfflineNavigationService::class.java).setAction(START)
                    .putExtra(EXTRA_REROUTE, allowReroute).putExtra(EXTRA_REVISION, expectedRevision))
            Unit
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, OfflineNavigationService::class.java))
        }

        /** Explicit microphone use may interrupt an announcement without stopping the journey. */
        fun interruptSpeech() {
            instance?.let { service -> service.main.post {
                if (service.active) service.stopSpeech(clearPending = true)
            } }
        }
    }

    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-offline-navigation").apply { isDaemon = true }
    }
    private val session = AtomicLong()
    private val pendingFix = AtomicReference<PendingFix?>()
    private val fixDrainScheduled = AtomicBoolean()
    private val pendingDelivery = AtomicReference<Delivery?>()
    private val deliveryScheduled = AtomicBoolean()
    @Volatile private var active = false
    @Volatile private var routeRevision = -1L
    @Volatile private var activeRoute: OfflineRoute? = null
    private var guidance: OfflineGuidance? = null // Created and used only on the route worker.
    private lateinit var store: OfflineRouteStore
    private lateinit var router: OfflineCarRouter
    private lateinit var locations: LocationManager
    private lateinit var audio: AudioManager
    private var originalDestination: OfflineCoordinate? = null
    private var allowReroute = false
    private var rerouting = false
    @Volatile private var routeSaving = false
    @Volatile private var gpsEnabled = false
    private var lastRerouteAttemptMillis: Long? = null
    private var lastReceivedFix: OfflineGuidance.Fix? = null
    private var generation = 0L
    private var arrived = false
    private var progress: OfflineRoute.Progress? = null
    private var baseStatus = "Navigation is off"
    private var routeNotice: String? = null
    private var routeNoticeUntil = 0L
    private var tts: TextToSpeech? = null
    private var speechInitialised = false
    private var speechUsable = false
    private var speechWarning: String? = null
    private var pendingSpeech: Announcement? = null // One latest instruction, never an unbounded queue.
    private var currentSpeechId: String? = null
    private var currentSpeechLease: Long? = null
    private var currentSpeechOffRoute = false
    private var utteranceSequence = 0L
    private var lastSpeechText: String? = null
    private var lastSpeechMillis = 0L
    private var focusRequest: AudioFocusRequest? = null
    private var focusHeld = false
    private var focusSequence = 0L
    private var destroyed = false

    private data class Delivery(val session: Long, val revision: Long,
        val fix: OfflineGuidance.Fix, val update: OfflineGuidance.Update)
    private data class PendingFix(val session: Long, val revision: Long, val fix: OfflineGuidance.Fix)
    private data class Announcement(val text: String, val session: Long, val revision: Long,
        val createdMillis: Long, val coordinate: OfflineCoordinate, val offRoute: Boolean,
        val manoeuvreIndex: Int?)

    private val attributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build()
    private val listener = object : LocationListener {
        override fun onLocationChanged(location: Location) {
            if (!active) return
            val coordinate = runCatching { OfflineCoordinate(location.latitude, location.longitude) }
                .getOrNull() ?: return
            val timestamp = location.elapsedRealtimeNanos / 1_000_000L
            val now = SystemClock.elapsedRealtime()
            if (timestamp <= 0L || timestamp > now ||
                timestamp <= (lastReceivedFix?.elapsedRealtimeMillis ?: 0L)) return
            val accuracy = if (location.hasAccuracy()) location.accuracy.toDouble() else Double.NaN
            val fix = OfflineGuidance.Fix(coordinate, accuracy, timestamp)
            lastReceivedFix = fix
            enqueueFix(fix)
        }

        override fun onProviderDisabled(provider: String) {
            if (provider == LocationManager.GPS_PROVIDER && active) {
                gpsEnabled = false
                stopSpeech(clearPending = true)
                publish("GPS is switched off; guidance is paused")
            }
        }

        override fun onProviderEnabled(provider: String) {
            if (provider == LocationManager.GPS_PROVIDER && active) {
                gpsEnabled = true
                publish("Waiting for an accurate GPS position")
            }
        }

        @Deprecated("Required by older Android location services")
        override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) = Unit
    }

    private val watchdog = object : Runnable {
        override fun run() {
            if (!active) return
            if (!hasPrecisePermission()) { finish("Precise location permission was removed"); return }
            if (!store.isCurrent(routeRevision)) { finish("The saved route changed; navigation stopped"); return }
            val fix = lastReceivedFix
            if (!gpsEnabled && activeRoute != null) publish("GPS is switched off; guidance is paused")
            else if (fix != null) enqueueFix(fix) // Re-evaluate freshness even when GPS stops delivering fixes.
            else if (activeRoute != null && !rerouting) publish("Waiting for an accurate GPS position")
            main.postDelayed(this, 2_000L)
        }
    }

    private val speechTick = object : Runnable {
        override fun run() {
            if (!active) return
            if (currentSpeechId != null && otherVoiceSpeaking()) stopSpeech(clearPending = true)
            pumpSpeech()
            main.postDelayed(this, 250L)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        store = OfflineRouteStore(File(filesDir, "offline-navigation/route.gpx"))
        router = OfflineCarRouter(this)
        locations = getSystemService(LOCATION_SERVICE) as LocationManager
        audio = getSystemService(AUDIO_SERVICE) as AudioManager
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(
            NotificationChannel(CHANNEL, "ORION offline navigation", NotificationManager.IMPORTANCE_LOW))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) { finish("Navigation stopped"); return START_NOT_STICKY }
        if (intent?.action != START || !intent.hasExtra(EXTRA_REVISION)) {
            finish("Open the map to start navigation"); return START_NOT_STICKY
        }
        if (!hasPrecisePermission()) { finish("Allow precise location before starting navigation"); return START_NOT_STICKY }
        val expectedRevision = intent.getLongExtra(EXTRA_REVISION, -1L)
        if (expectedRevision < 0L || !store.isCurrent(expectedRevision)) {
            finish("The saved route changed; select the latest route"); return START_NOT_STICKY
        }
        if (active && routeRevision == expectedRevision) return START_NOT_STICKY
        resetSession()
        routeRevision = expectedRevision
        allowReroute = intent.getBooleanExtra(EXTRA_REROUTE, false)
        val sessionAllowsReroute = allowReroute
        active = true
        val token = session.incrementAndGet()
        try {
            val notification = notification("Loading the saved route")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
            else startForeground(NOTIFICATION_ID, notification)
        } catch (_: RuntimeException) {
            finish("Android could not start navigation; open the map and check location permission")
            return START_NOT_STICKY
        }
        publish("Loading the saved route")
        initialiseSpeech()
        main.post(watchdog) // Also detects route supersession whilst restoration is still running.
        executeWorker {
            val result = runCatching {
                val (route, revision) = store.readSnapshot()
                if (!isCurrent(token, expectedRevision) || revision != expectedRevision)
                    throw IOException("The saved route changed during restoration")
                val restored = route ?: throw IOException("Calculate or import a route before starting navigation")
                val engine = OfflineGuidance(restored, sessionAllowsReroute)
                if (!isCurrent(token, expectedRevision)) throw IOException("Route restoration was superseded")
                guidance = engine
                restored
            }
            main.post {
                if (!ensureCurrent(token, expectedRevision)) return@post
                result.fold(onSuccess = { route ->
                    activeRoute = route
                    originalDestination = route.points.last()
                    generation = generations.incrementAndGet()
                    publish("Waiting for an accurate GPS position")
                    requestLocations()
                    main.removeCallbacks(watchdog)
                    main.post(watchdog)
                    main.removeCallbacks(speechTick)
                    main.post(speechTick)
                }, onFailure = { finish(it.message?.take(180) ?: "The saved route could not be opened") })
            }
        }
        return START_NOT_STICKY // No restart, boot receiver or automatic route restoration.
    }

    private fun hasPrecisePermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED

    private fun requestLocations() {
        try {
            gpsEnabled = locations.isProviderEnabled(LocationManager.GPS_PROVIDER)
            if (!gpsEnabled) {
                publish("GPS is switched off; guidance is paused")
            }
            // GPS is independent of mobile data. Never substitute a network-provider position.
            locations.requestLocationUpdates(LocationManager.GPS_PROVIDER, 1_000L, 2f, listener, Looper.getMainLooper())
        } catch (_: SecurityException) { finish("Allow precise location before starting navigation") }
        catch (_: IllegalArgumentException) { finish("This phone does not provide GPS navigation") }
    }

    private fun enqueueFix(fix: OfflineGuidance.Fix) {
        if (!active || !gpsEnabled || routeSaving) return
        val next = PendingFix(session.get(), routeRevision, fix)
        pendingFix.updateAndGet { previous ->
            if (previous != null && previous.session == next.session && previous.revision == next.revision &&
                previous.fix.elapsedRealtimeMillis > next.fix.elapsedRealtimeMillis) previous else next
        }
        scheduleFixDrain()
    }

    private fun scheduleFixDrain() {
        // Do not consume another prompt before the main thread has accepted this delivery.
        if (!active || deliveryScheduled.get()) return
        if (!fixDrainScheduled.compareAndSet(false, true)) return
        executeWorker {
            try {
                val next = pendingFix.getAndSet(null)
                if (next != null && gpsEnabled && !routeSaving && isCurrent(next.session, next.revision)) {
                    val engine = guidance
                    if (engine == null) return@executeWorker
                    try {
                        val update = engine.update(next.fix, SystemClock.elapsedRealtime())
                        if (isCurrent(next.session, next.revision))
                            deliver(Delivery(next.session, next.revision, next.fix, update))
                    } catch (_: Exception) {
                        main.post { if (ensureCurrent(next.session, next.revision))
                            finish("The saved route could not be followed") }
                    }
                }
            } finally {
                fixDrainScheduled.set(false)
                if (active && pendingFix.get() != null) scheduleFixDrain()
            }
        }
    }

    /** The worker also has one bounded delivery slot when the UI thread is busy. */
    private fun deliver(delivery: Delivery) {
        pendingDelivery.set(delivery)
        scheduleDelivery()
    }

    private fun scheduleDelivery() {
        if (!deliveryScheduled.compareAndSet(false, true)) return
        main.post {
            try { pendingDelivery.getAndSet(null)?.let { applyUpdate(it) } }
            finally {
                deliveryScheduled.set(false)
                if (active && pendingDelivery.get() != null) scheduleDelivery()
                else if (active && pendingFix.get() != null) scheduleFixDrain()
            }
        }
    }

    private fun applyUpdate(delivery: Delivery) {
        if (!gpsEnabled || routeSaving || !ensureCurrent(delivery.session, delivery.revision)) return
        val update = delivery.update
        progress = update.progress
        if (update.progress == null) stopSpeech(clearPending = true)
        val outsideCorridor = update.progress?.distanceFromRouteMetres?.let {
            it > maxOf(30.0, delivery.fix.accuracyMetres)
        } == true
        if (outsideCorridor && (currentSpeechId != null && !currentSpeechOffRoute || pendingSpeech?.offRoute == false))
            stopSpeech(clearPending = true)
        publish(if (rerouting) "Recalculating the offline route" else update.status,
            delivery.fix.coordinate, delivery.fix.elapsedRealtimeMillis)
        if (update.speech != null) offerSpeech(update.speech, delivery.fix, outsideCorridor)
        if (update.arrived && !arrived) {
            arrived = true
            // Keep the last arrival direction long enough to finish; never route past arrival.
            runCatching { locations.removeUpdates(listener) }
            main.removeCallbacks(watchdog)
            main.postDelayed({ if (active && session.get() == delivery.session) finish("Arrived at the destination") }, 12_000L)
        } else if (!arrived && update.requestReroute) requestReroute(delivery.fix)
    }

    private fun requestReroute(fix: OfflineGuidance.Fix) {
        val now = SystemClock.elapsedRealtime()
        val destination = originalDestination ?: return
        if (!active || arrived || !allowReroute || rerouting ||
            now - fix.elapsedRealtimeMillis !in 0L..10_000L ||
            !fix.accuracyMetres.isFinite() || fix.accuracyMetres !in 0.0..35.0 ||
            lastRerouteAttemptMillis?.let { now - it < REROUTE_COOLDOWN_MILLIS } == true) return
        val token = session.get()
        val revision = routeRevision
        val sessionAllowsReroute = allowReroute
        if (!isCurrent(token, revision)) return
        lastRerouteAttemptMillis = now
        rerouting = true
        if (!currentSpeechOffRoute && pendingSpeech?.offRoute != true) stopSpeech(clearPending = true)
        publish("Recalculating the offline route")
        router.calculate(fix.coordinate, destination) { result ->
            if (!ensureCurrent(token, revision)) return@calculate
            result.fold(onSuccess = { xml ->
                // Reserve on the main thread, then save on the worker. A newer map request wins.
                val nextRevision = store.beginChangeIfCurrent(revision)
                if (nextRevision == null) { finish("The saved route changed; navigation stopped"); return@fold }
                stopSpeech(clearPending = true)
                routeSaving = true
                routeRevision = nextRevision
                pendingFix.set(null)
                executeWorker {
                    val saved = runCatching {
                        val parsed = OfflineRouteGpx.parse(xml)
                        if (OfflineRoute.distance(parsed.points.last(), destination) > 200.0)
                            throw IOException("The recalculated route did not reach the saved destination")
                        if (!isCurrent(token, nextRevision)) throw IOException("Route calculation was superseded")
                        val route = store.save(xml, nextRevision) { isCurrent(token, nextRevision) }
                        val engine = OfflineGuidance(route, sessionAllowsReroute)
                        if (isCurrent(token, nextRevision)) guidance = engine
                        route
                    }
                    main.post {
                        if (!ensureCurrent(token, nextRevision)) return@post
                        rerouting = false
                        routeSaving = false
                        saved.fold(onSuccess = { route ->
                            activeRoute = route
                            progress = null
                            generation = generations.incrementAndGet()
                            publish("Offline route recalculated")
                            lastReceivedFix?.let { enqueueFix(it) }
                        }, onFailure = { rerouteFailed(it.message); lastReceivedFix?.let { enqueueFix(it) } })
                    }
                }
            }, onFailure = { rerouting = false; rerouteFailed(it.message) })
        }
    }

    private fun rerouteFailed(message: String?) {
        routeNotice = "Recalculation unavailable: ${message?.take(160) ?: "check BRouter's downloaded regions"}"
        routeNoticeUntil = SystemClock.elapsedRealtime() + 10_000L
        publish("Continue using the saved route")
    }

    private fun initialiseSpeech() {
        if (tts != null || destroyed) return
        tts = TextToSpeech(this) { result -> main.post {
            val engine = tts ?: return@post
            if (destroyed) return@post
            speechInitialised = true
            val voice = runCatching { engine.voices.orEmpty().filter {
                !it.isNetworkConnectionRequired && it.locale.language == "en" && it.locale.country == "GB" &&
                    !it.features.orEmpty().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)
            }.sortedWith(compareByDescending<android.speech.tts.Voice> { it.quality }
                .thenBy { it.latency }.thenBy { it.name }).firstOrNull() }.getOrNull()
            speechUsable = result == TextToSpeech.SUCCESS && voice != null &&
                engine.setVoice(voice) == TextToSpeech.SUCCESS && engine.setAudioAttributes(attributes) == TextToSpeech.SUCCESS
            if (!speechUsable) {
                speechWarning = "Offline UK speech is unavailable; download a UK voice in the phone's speech settings"
                pendingSpeech = null
            } else {
                speechWarning = null
                engine.setSpeechRate(1.06f)
                engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                    override fun onStart(utteranceId: String) = Unit
                    override fun onDone(utteranceId: String) { main.post { completeSpeech(utteranceId, false) } }
                    @Deprecated("Required by older Android speech engines")
                    override fun onError(utteranceId: String) { main.post { completeSpeech(utteranceId, true) } }
                    override fun onError(utteranceId: String, errorCode: Int) { main.post { completeSpeech(utteranceId, true) } }
                    override fun onStop(utteranceId: String, interrupted: Boolean) { main.post { completeSpeech(utteranceId, false) } }
                })
            }
            if (active) { publish(baseStatus); pumpSpeech() }
        } }
        main.postDelayed({
            if (active && !speechInitialised) {
                speechWarning = "Offline UK speech did not initialise; visual guidance remains available"
                pendingSpeech = null
                publish(baseStatus)
            }
        }, 10_000L)
    }

    private fun offerSpeech(text: String, fix: OfflineGuidance.Fix, offRoute: Boolean) {
        val clean = text.trim().replace(Regex("\\s+"), " ").take(200)
        if (clean.isEmpty() || !active || speechInitialised && !speechUsable) return
        val now = SystemClock.elapsedRealtime()
        if (clean == lastSpeechText && now - lastSpeechMillis < 15_000L) return
        pendingSpeech = Announcement(clean, session.get(), routeRevision, now, fix.coordinate,
            offRoute, progress?.nextManoeuvre?.pointIndex)
        pumpSpeech()
    }

    private fun otherVoiceSpeaking(): Boolean = PhoneVoice.anySpeaking() || RemoteSpeech.anySpeaking()

    @Suppress("DEPRECATION") // Explicit embedded synthesis supplements the selected non-network voice.
    private fun pumpSpeech() {
        val announcement = pendingSpeech ?: return
        val now = SystemClock.elapsedRealtime()
        if (!isCurrent(announcement.session, announcement.revision) ||
            now - announcement.createdMillis > SPEECH_MAX_WAIT_MILLIS) { pendingSpeech = null; return }
        val fix = lastReceivedFix
        if (!gpsEnabled || progress == null || fix == null || now - fix.elapsedRealtimeMillis !in 0L..10_000L ||
            !fix.accuracyMetres.isFinite() || fix.accuracyMetres !in 0.0..50.0) {
            pendingSpeech = null
            return
        }
        if (!announcement.offRoute &&
            (progress?.nextManoeuvre?.pointIndex != announcement.manoeuvreIndex ||
                OfflineRoute.distance(announcement.coordinate, fix.coordinate) > 30.0 ||
                progress!!.distanceFromRouteMetres > maxOf(30.0, fix.accuracyMetres))) {
            pendingSpeech = null // Never read an old turn or distance after passing its position.
            return
        }
        if (!speechUsable || currentSpeechId != null || otherVoiceSpeaking()) return
        val focusToken = ++focusSequence
        val listener = AudioManager.OnAudioFocusChangeListener { change ->
            if (focusSequence == focusToken && focusHeld &&
                (change == AudioManager.AUDIOFOCUS_LOSS || change == AudioManager.AUDIOFOCUS_LOSS_TRANSIENT))
                stopSpeech(clearPending = true)
        }
        val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
            .setAudioAttributes(attributes).setAcceptsDelayedFocusGain(false)
            .setOnAudioFocusChangeListener(listener, main).build()
        if (runCatching { audio.requestAudioFocus(request) }.getOrNull() != AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
            runCatching { audio.abandonAudioFocusRequest(request) }
            pendingSpeech = null
            return
        }
        focusRequest = request
        focusHeld = true
        val id = "orion-navigation-${session.get()}-${++utteranceSequence}"
        currentSpeechId = id
        currentSpeechOffRoute = announcement.offRoute
        currentSpeechLease = NavigationAudioState.begin()
        pendingSpeech = null
        lastSpeechText = announcement.text
        lastSpeechMillis = now
        val parameters = Bundle().apply { putString(TextToSpeech.Engine.KEY_FEATURE_EMBEDDED_SYNTHESIS, "true") }
        if (tts?.speak(announcement.text, TextToSpeech.QUEUE_FLUSH, parameters, id) != TextToSpeech.SUCCESS) {
            completeSpeech(id, true)
        } else main.postDelayed({
            if (currentSpeechId == id) {
                stopSpeech(clearPending = true)
                speechWarning = "Offline speech did not finish; visual guidance remains available"
                speechUsable = false
                if (active) publish(baseStatus)
            }
        }, 20_000L)
    }

    private fun completeSpeech(id: String, failed: Boolean) {
        if (currentSpeechId != id) return
        currentSpeechId = null
        currentSpeechOffRoute = false
        currentSpeechLease?.let { NavigationAudioState.end(it, SystemClock.elapsedRealtime()) }
        currentSpeechLease = null
        releaseFocus()
        if (failed) {
            speechUsable = false
            speechWarning = "Offline speech failed; visual guidance remains available"
            pendingSpeech = null
            if (active) publish(baseStatus)
        } else pumpSpeech()
    }

    private fun stopSpeech(clearPending: Boolean) {
        if (clearPending) pendingSpeech = null
        val id = currentSpeechId
        if (id != null) {
            tts?.stop()
            completeSpeech(id, false)
        }
        releaseFocus()
    }

    private fun releaseFocus() {
        ++focusSequence
        if (focusHeld) focusRequest?.let { runCatching { audio.abandonAudioFocusRequest(it) } }
        focusHeld = false
        focusRequest = null
    }

    private fun isCurrent(token: Long, revision: Long): Boolean =
        active && !destroyed && !Thread.currentThread().isInterrupted && session.get() == token &&
            routeRevision == revision && store.isCurrent(revision)

    /** Main-thread callbacks stop a superseded journey rather than leaving a loading service. */
    private fun ensureCurrent(token: Long, revision: Long): Boolean {
        if (!active || destroyed || session.get() != token || routeRevision != revision) return false
        if (!store.isCurrent(revision)) { finish("The saved route changed; navigation stopped"); return false }
        return true
    }

    private fun publish(message: String, coordinate: OfflineCoordinate? = snapshot.coordinate,
        fixMillis: Long? = snapshot.lastFixElapsedRealtimeMillis) {
        baseStatus = message
        val notice = routeNotice?.takeIf { SystemClock.elapsedRealtime() < routeNoticeUntil }
        val displayed = if (notice == null) message else "$message. $notice"
        snapshot = Snapshot(active, displayed, activeRoute, routeRevision.takeIf { it >= 0L }, generation,
            coordinate, progress, fixMillis, speechUsable, speechWarning, allowReroute)
        if (active) (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .notify(NOTIFICATION_ID, notification(displayed))
    }

    private fun notification(message: String): Notification {
        val open = PendingIntent.getActivity(this, NOTIFICATION_ID,
            Intent(this, OfflineMapActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val stop = PendingIntent.getService(this, NOTIFICATION_ID + 1,
            Intent(this, OfflineNavigationService::class.java).setAction(STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL).setSmallIcon(android.R.drawable.ic_menu_mylocation)
            .setContentTitle("ORION navigation").setContentText(message.take(220))
            .setContentIntent(open).setOngoing(true).setOnlyAlertOnce(true)
            .setCategory(if (Build.VERSION.SDK_INT >= 28) Notification.CATEGORY_NAVIGATION else Notification.CATEGORY_SERVICE)
            .addAction(Notification.Action.Builder(android.graphics.drawable.Icon.createWithResource(this,
                android.R.drawable.ic_media_pause), "Stop", stop).build())
            .build()
    }

    private fun executeWorker(task: () -> Unit) {
        try { worker.execute(task) }
        catch (_: RejectedExecutionException) { if (active) main.post { finish("Navigation stopped") } }
    }

    private fun resetSession() {
        active = false
        session.incrementAndGet()
        main.removeCallbacks(watchdog)
        main.removeCallbacks(speechTick)
        runCatching { locations.removeUpdates(listener) }
        router.cancel()
        stopSpeech(clearPending = true)
        pendingFix.set(null)
        pendingDelivery.set(null)
        lastReceivedFix = null
        progress = null
        arrived = false
        activeRoute = null
        originalDestination = null
        rerouting = false
        routeSaving = false
        gpsEnabled = false
        routeNotice = null
        routeNoticeUntil = 0L
        lastSpeechText = null
        // Last attempt time deliberately survives a successful reroute and an explicit restart.
    }

    private fun finish(message: String) {
        resetSession()
        publish(message, null, null)
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        val stoppedUnexpectedly = active
        resetSession()
        destroyed = true
        main.removeCallbacksAndMessages(null)
        router.close()
        worker.shutdownNow()
        tts?.stop()
        tts?.shutdown()
        tts = null
        NavigationAudioState.clear(SystemClock.elapsedRealtime())
        if (instance === this) {
            instance = null
            snapshot = snapshot.copy(running = false, status = if (stoppedUnexpectedly) "Navigation stopped" else snapshot.status,
                route = null, coordinate = null, progress = null)
        }
        super.onDestroy()
    }
}
