package com.orion.companion

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.audiofx.AcousticEchoCanceler
import android.media.MediaRecorder
import android.net.ConnectivityManager
import android.net.Network
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import androidx.core.content.ContextCompat
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min
import kotlin.random.Random

/** User-started microphone service. The ongoing notification stops recording. */
class OrionAudioService : Service() {
    companion object {
        private val events = java.util.ArrayDeque<String>()
        @Volatile var status = "{\"state\":\"off\",\"text\":\"Background listening is off\"}"
            private set
        @Volatile var level = 0.0
            private set
        @Volatile var captureActive = false
            private set
        @Volatile var echoCancellation = false
            private set
        fun drainEvents(): List<String> = synchronized(events) {
            events.toList().also { events.clear() }
        }
        private const val START = "com.orion.companion.AUDIO_START"
        private const val STOP = "com.orion.companion.AUDIO_STOP"
        private const val CHANNEL = "orion_audio"
        private const val NOTIFICATION_ID = 403
        @Volatile var running = false
            private set

        fun start(context: Context) {
            ContextCompat.startForegroundService(context,
                Intent(context, OrionAudioService::class.java).setAction(START))
        }
        fun stop(context: Context) {
            // The user may stop during foreground-service startup, before
            // `running` becomes true. Stop the pending service as well.
            context.stopService(Intent(context, OrionAudioService::class.java))
        }
    }

    private val worker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-background-audio").apply { isDaemon = true }
    }
    private val client by lazy { pinnedHttpClient(applicationContext) }
    @Volatile private var active = false
    @Volatile private var currentState = "off"
    @Volatile private var socket: WebSocket? = null
    @Volatile private var recorder: AudioRecord? = null
    private var tts: RemoteSpeech? = null
    private var wakeLock: PowerManager.WakeLock? = null
    @Volatile private var defaultNetwork: Network? = null
    private var networkCallbackRegistered = false
    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) {
            val previous = defaultNetwork
            defaultNetwork = network
            if (previous != null && previous != network) {
                val old = socket
                socket = null
                awaitingReply = false
                old?.cancel()
            }
        }
        override fun onLost(network: Network) {
            if (defaultNetwork == network) {
                defaultNetwork = null
                val old = socket
                socket = null
                awaitingReply = false
                old?.cancel()
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
            .createNotificationChannel(NotificationChannel(CHANNEL, "ORION listening",
                NotificationManager.IMPORTANCE_LOW))
        tts = RemoteSpeech(this, onSpeaking = { speaking ->
            publish(if (speaking) "speaking" else "listening",
                if (speaking) "ORION is speaking. Say ‘ORION stop’ or use Stop speaking."
                else "Say ‘ORION’ followed by your next request.")
        }, onVoice = { voice ->
            if (voice.startsWith("Voice ")) publish("error", voice)
        })
        try {
            (getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager)
                .registerDefaultNetworkCallback(networkCallback)
            networkCallbackRegistered = true
        } catch (_: Exception) { }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        if (active || ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED || Prefs.credentials(this) == null) {
            if (!active) stopSelf()
            return START_NOT_STICKY
        }
        val stopIntent = PendingIntent.getService(this, 0,
            Intent(this, OrionAudioService::class.java).setAction(STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val note = Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_orion)
            .setContentTitle("ORION is listening")
            .setContentText("Say ‘ORION’ followed by a request. Tap Stop to end listening.")
            .setOngoing(true)
            .addAction(Notification.Action.Builder(null, "Stop", stopIntent).build())
            .build()
        try {
            if (Build.VERSION.SDK_INT >= 30) {
                startForeground(NOTIFICATION_ID, note,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE)
            } else startForeground(NOTIFICATION_ID, note)
        } catch (_: Exception) {
            // The system may reject foreground microphone access after the
            // Activity checked permission, for example during a handoff.
            publish("error", "Android could not start background microphone access. Open ORION and try again.")
            stopSelf()
            return START_NOT_STICKY
        }
        active = true
        running = true
        wakeLock = (getSystemService(POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ORION:BackgroundAudio").apply {
                setReferenceCounted(false)
                acquire(10 * 60_000L)
            }
        worker.execute(::runAudio)
        return START_NOT_STICKY
    }

    private fun runAudio() {
        // Permission can be revoked after the service starts or while a
        // reconnect is pending. Check again on the recording thread.
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED) {
            publish("error", "Allow ORION's microphone permission in Android settings.")
            stopSelf()
            return
        }
        val minBuffer = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT)
        if (minBuffer <= 0) { publish("error", "The phone microphone is unavailable."); stopSelf(); return }
        val record = try {
            AudioRecord(if (AcousticEchoCanceler.isAvailable()) MediaRecorder.AudioSource.VOICE_COMMUNICATION
                else MediaRecorder.AudioSource.VOICE_RECOGNITION, 16000,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuffer * 2, 4096))
        } catch (_: Exception) {
            // A permission change between the check and construction must not
            // leave a live service with no recording worker.
            publish("error", "The phone microphone could not open. Close other recording apps and retry.")
            stopSelf()
            return
        }
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            publish("error", "The phone microphone did not initialise.")
            record.release(); stopSelf(); return
        }
        recorder = record
        val echo = runCatching {
            if (AcousticEchoCanceler.isAvailable()) AcousticEchoCanceler.create(record.audioSessionId)
                ?.apply { enabled = true } else null
        }.getOrNull()
        echoCancellation = runCatching { echo?.enabled == true }.getOrDefault(false)
        try {
            record.startRecording()
            check(record.recordingState == AudioRecord.RECORDSTATE_RECORDING)
            captureActive = true
            val frame = ByteArray(640) // 20 ms at 16 kHz, mono PCM16.
            val segmenter = PcmSpeechSegmenter()
            var retry = 0
            var renewed = SystemClock.elapsedRealtime()
            var lastHealth = renewed
            var waitingSince = 0L
            var captureWasSpeaking = false
            while (active) {
                val now = SystemClock.elapsedRealtime()
                if (now - renewed > 5 * 60_000L) {
                    wakeLock?.acquire(10 * 60_000L)
                    renewed = now
                }
                if (socket == null) {
                    segmenter.reset()
                    waitingSince = 0L
                    if (openSocket()) retry = 0
                    else {
                        retry = min(retry + 1, 5)
                        Thread.sleep(Random.nextLong(500L, (1000L shl retry) + 1))
                        continue
                    }
                }
                val count = record.read(frame, 0, frame.size)
                if (count < 0) error("Microphone read failed")
                if (count == 0) continue
                if (now - lastHealth >= 15_000) {
                    socket?.send("{\"type\":\"capture_status\",\"capturing\":true}")
                    lastHealth = now
                }
                if (NavigationAudioState.captureBlocked(SystemClock.elapsedRealtime())) {
                    segmenter.reset(); level = 0.0
                    // Keep the existing reply/socket state whilst directions use the loudspeaker.
                    continue
                }
                val interruptOnly = RemoteSpeech.anySpeaking()
                // Never combine a reply-interruption clip with a normal command.
                if (interruptOnly != captureWasSpeaking) {
                    segmenter.reset()
                    captureWasSpeaking = interruptOnly
                }
                if (interruptOnly && !echoCancellation) {
                    segmenter.reset(); level = 0.0
                    continue
                }
                val bytes = segmenter.accept(frame, count)
                level = segmenter.level
                if (waitingSince != 0L) {
                    segmenter.reset()
                    if (awaitingReply && now - waitingSince < 210_000) continue
                    awaitingReply = false
                    waitingSince = 0L
                    continue
                }
                if (bytes != null) {
                    val ws = socket
                    if (ws != null && ws.queueSize() < 500_000) {
                        var sent = true
                        for (offset in bytes.indices step 16000) {
                            val end = min(offset + 16000, bytes.size)
                            sent = ws.send(bytes.copyOfRange(offset, end).toByteString()) && sent
                        }
                        if (sent && ws.send(JSONObject().put("type", "commit")
                                .put("interrupt_only", interruptOnly).toString())) {
                            awaitingReply = true
                            waitingSince = now
                            publish("transcribing", "Turning your speech into words…")
                        } else { socket = null; ws.cancel() }
                    }
                }
                if (!awaitingReply) waitingSince = 0L
            }
        } catch (_: Exception) {
            if (active) publish("error", "Background capture stopped. Check microphone permission and try LISTEN again.")
        } finally {
            try { record.stop() } catch (_: Exception) { }
            record.release()
            echo?.release()
            echoCancellation = false
            recorder = null
            captureActive = false
            level = 0.0
            socket?.cancel()
            socket = null
            if (active) stopSelf()
        }
    }

    @Volatile private var awaitingReply = false

    private fun openSocket(): Boolean {
        val credentials = Prefs.credentials(this) ?: return false
        publish("connecting", "Connecting the background microphone to the PC…")
        for (baseText in (listOfNotNull(Prefs.baseUrl(this)) + Prefs.candidateUrls(this)).distinct()) {
            val base = baseText.toHttpUrlOrNull() ?: continue
            if (base.scheme != "https") continue
            val tokenUrl = base.newBuilder().encodedPath("/v1/auth/token").query(null).build()
            val body = JSONObject().put("device_id", credentials.first)
                .put("refresh_token", credentials.second).toString()
                .toRequestBody("application/json; charset=utf-8".toMediaType())
            val token = try {
                client.newCall(Request.Builder().url(tokenUrl).post(body).build()).execute().use {
                    if (!it.isSuccessful) null
                    else JSONObject(it.body?.string().orEmpty()).optString("access_token")
                }
            } catch (_: Exception) { null } ?: continue
            if (token.isBlank()) continue
            val wsUrl = base.newBuilder().encodedPath("/v1/audio/stream").query(null).build()
            val opened = java.util.concurrent.CountDownLatch(1)
            val handshake = Any()
            val accepting = AtomicBoolean(true)
            val accepted = AtomicBoolean(false)
            val ws = client.newWebSocket(Request.Builder().url(wsUrl)
                .header("Authorization", "Bearer $token").build(), object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    synchronized(handshake) {
                        if (!active || !accepting.get()) {
                            webSocket.cancel(); opened.countDown(); return
                        }
                        socket = webSocket; accepted.set(true); opened.countDown()
                    }
                    if (active && socket === webSocket)
                        publish("listening", "Say ‘ORION’ followed by your request. The microphone button also accepts dictation.")
                }
                override fun onMessage(webSocket: WebSocket, text: String) {
                    if (!active || socket !== webSocket) return
                    val message = runCatching { JSONObject(text) }.getOrNull() ?: return
                    if (message.optString("type") == "interrupt") {
                        awaitingReply = false
                        RemoteSpeech.stopAll()
                        publish("listening", "Speech stopped. Say ‘ORION’ followed by your next request.")
                    } else if (message.optString("type") == "tts_text") {
                        awaitingReply = false
                        val spoken = message.optString("text")
                        publish("reply", spoken, message)
                        tts?.speak(spoken)
                    } else if (message.optString("type") == "transcript") {
                        publish("transcript", message.optString("text").take(4000))
                    } else if (message.optString("type") == "status") {
                        awaitingReply = false
                        publish("status", message.optString("message").take(300))
                    } else if (message.optString("type") == "error") {
                        awaitingReply = false
                        publish("error", message.optString("message", "Background voice request failed").take(180))
                        android.util.Log.w("OrionAudio", "Background voice request failed")
                    }
                }
                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    if (socket === webSocket) {
                        socket = null
                        awaitingReply = false
                        if (active) publish("connecting", "Background audio disconnected. Reconnecting…")
                    }
                    opened.countDown()
                }
                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    if (socket === webSocket) { socket = null; awaitingReply = false }
                    opened.countDown()
                }
                override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                    webSocket.close(code, null)
                    if (socket === webSocket) { socket = null; awaitingReply = false }
                }
            })
            opened.await(8, TimeUnit.SECONDS)
            synchronized(handshake) {
                accepting.set(false)
                if (accepted.get() && active && socket === ws) return true
                if (socket === ws) socket = null
            }
            ws.cancel()
        }
        return false
    }

    private fun publish(state: String, text: String, details: JSONObject? = null) {
        val payload = JSONObject().put("state", state).put("text", text)
        if (details != null) for (key in listOf("widgets", "user_text", "user_hid", "hid")) {
            if (details.has(key)) payload.put(key, details.get(key))
        }
        val event = payload.toString()
        if (state != "transcript" && state != "reply") {
            currentState = state
            status = event
        }
        synchronized(events) {
            events.addLast(event)
            // Bound both message count and retained rich-card data when closed.
            while (events.size > 64 || events.sumOf { it.length } > 8_388_608) events.removeFirst()
        }
    }

    override fun onDestroy() {
        active = false
        running = false
        captureActive = false
        level = 0.0
        wakeLock?.let { if (it.isHeld) it.release() }
        wakeLock = null
        if (currentState != "error")
            publish("off", "Background listening is off. Tap the microphone to dictate.")
        try { recorder?.stop() } catch (_: Exception) { }
        val oldSocket = socket
        socket = null
        worker.shutdownNow()
        tts?.close()
        tts = null
        closeHttpResources(client) { oldSocket?.cancel() }
        if (networkCallbackRegistered) {
            try { (getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager)
                .unregisterNetworkCallback(networkCallback) } catch (_: Exception) { }
        }
        super.onDestroy()
    }
}
