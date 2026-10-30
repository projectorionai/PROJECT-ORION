package com.orion.companion

import android.content.Context
import okhttp3.Call
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * The PC's face, live, for the phone console: ORION's state, speech level and
 * emotion from the paired PC's /api/events stream, and his current design
 * and colours from the token answer. Home Wi-Fi first (the endpoints are
 * ordered LAN, then Tailscale); if the PC's LAN address moved, it is found
 * again by its pinned certificate. While nothing answers, [connected] is
 * false and the console's face follows the phone itself.
 *
 * The access token travels in the Authorization header only, over the same
 * pinned/CA-checked HTTPS client as every other PC call.
 */
internal class PcFaceLink(
    context: Context,
    private val onEmotion: (String, Double) -> Unit,
    private val onStyle: (FaceStyle) -> Unit,
) {
    private val app = context.applicationContext
    private val http = pinnedHttpClient(app).newBuilder()
        .connectTimeout(4, TimeUnit.SECONDS)
        // The PC writes a keep-alive comment well inside this; silence this
        // long means the link is gone.
        .readTimeout(45, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false).build()
    @Volatile private var stopped = false
    @Volatile private var call: Call? = null
    private var thread: Thread? = null

    @Volatile var connected = false
        private set
    @Volatile var state = "STANDBY"
        private set
    @Volatile var speaking = false
        private set
    @Volatile var amplitude = 0f
        private set
    /** "home Wi-Fi" or "the internet", for the status line. */
    @Volatile var route = ""
        private set

    fun start() {
        if (thread != null) return
        thread = Thread({ loop() }, "orion-pc-face").apply { isDaemon = true; start() }
    }

    fun stop() {
        stopped = true
        connected = false
        call?.cancel()
        thread?.interrupt()
        closeHttpResources(http)
    }

    private fun loop() {
        var backoff = 5_000L
        while (!stopped) {
            val started = System.currentTimeMillis()
            val streamed = runCatching { once() }.getOrDefault(false)
            connected = false
            speaking = false
            amplitude = 0f
            if (stopped) return
            if (!streamed && LanDiscovery.rediscover(app) != null) continue
            // A stream that ran a while ended normally (token lifetime):
            // reconnect at once. A failure backs off, gently on the PC.
            backoff = if (streamed && System.currentTimeMillis() - started > 60_000L) 0L
                else (backoff * 2).coerceIn(10_000L, 120_000L)
            if (backoff > 0) try { Thread.sleep(backoff) } catch (_: InterruptedException) { return }
        }
    }

    /** One token exchange and one event stream. True if the stream opened. */
    private fun once(): Boolean {
        val credentials = Prefs.credentials(app) ?: return false
        for (route in Prefs.candidateUrls(app).take(4)) {
            if (stopped) return false
            val base = Prefs.canonicalHttpsOrigin(route)?.toHttpUrlOrNull() ?: continue
            val token = try {
                val body = JSONObject().put("device_id", credentials.first)
                    .put("refresh_token", credentials.second).toString()
                    .toRequestBody("application/json; charset=utf-8".toMediaType())
                val request = Request.Builder().url(base.newBuilder().encodedPath("/v1/auth/token").build())
                    .post(body).build()
                http.newCall(request).also { call = it }.execute().use { response ->
                    val source = response.body?.source() ?: return@use null
                    if (!response.isSuccessful || source.request(65_537)) return@use null
                    val data = runCatching { JSONObject(source.readUtf8()) }.getOrNull() ?: return@use null
                    data.optJSONArray("endpoints")?.let { Prefs.saveEndpoints(app, it.toString()) }
                    FaceStyle.parse(data.optJSONObject("face"))?.let(onStyle)
                    data.optString("access_token").takeIf { it.length in 1..1024 }
                }
            } catch (_: Exception) { null } ?: continue
            val events = Request.Builder().url(base.newBuilder().encodedPath("/api/events").build())
                .header("Authorization", "Bearer $token").header("Accept", "text/event-stream").build()
            try {
                http.newCall(events).also { call = it }.execute().use { response ->
                    if (!response.isSuccessful) return@use
                    val source = response.body?.source() ?: return@use
                    connected = true
                    this.route = if (LanTrust.isPrivateLanHost(base.host)) "home Wi-Fi" else "the internet"
                    while (!stopped) {
                        val line = source.readUtf8Line() ?: break
                        if (line.length > 8_192 || !line.startsWith("data:")) continue
                        handle(line.removePrefix("data:").trim())
                    }
                }
                if (connected) return true
            } catch (_: Exception) {
                if (connected) return true
            }
        }
        return false
    }

    private fun handle(text: String) {
        val event = runCatching { JSONObject(text) }.getOrNull() ?: return
        when (event.optString("type")) {
            "state" -> state = PhoneAvatarPolicy.faceState(event.optString("value"))
            "speaking" -> { speaking = event.optBoolean("value"); if (!speaking) amplitude = 0f }
            "amplitude" -> amplitude = event.optDouble("value", 0.0).toFloat().coerceIn(0f, 1f)
            "emotion" -> {
                val name = event.optString("name").ifBlank {
                    event.optJSONObject("params")?.optString("name").orEmpty()
                }
                val intensity = event.optJSONObject("params")?.optDouble("intensity", 1.0) ?: 1.0
                if (name.matches(Regex("[a-z_ ]{1,40}"))) onEmotion(name, intensity)
            }
            // Approvals, thoughts and actions on this stream belong to Desktop
            // mode; the face ignores them.
        }
    }
}
