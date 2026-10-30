package com.orion.companion

import android.content.Context
import android.os.SystemClock
import okhttp3.Call
import okhttp3.Callback
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledFuture
import java.util.concurrent.TimeUnit
import kotlin.math.min
import kotlin.random.Random

/**
 * Pairing-authenticated phone uplink and nonblocking JPEG sink. The Activity
 * owns the connection for command proposals and telemetry. The projection
 * service sends frames only while its user-authorized capture is active.
 * Incoming JSON is delivered to a listener; this class never executes an action.
 */
object OrionFrameTransport {
    interface Listener {
        /** Callbacks arrive on an OkHttp or transport worker thread. */
        fun onConnectionChanged(connected: Boolean, latencyMs: Long?) {}
        fun onServerMessage(message: JSONObject) {}
        fun onPairingRequired() {}
        /**
         * A proposal from ORION, not permission to execute it. The UI must show
         * the parameters and obtain local, per-command approval before invoking
         * any AccessibilityService gesture or external action. On failure or
         * denial call sendCommandError with this request's id.
         */
        fun onCommandRequest(request: JSONObject) {}
    }

    private val lock = Any()
    @Volatile private var uplink: OrionUplinkClient? = null
    @Volatile private var listener: Listener? = null

    fun setListener(value: Listener?) { listener = value }

    /** Open the paired command channel; safe to call more than once. */
    fun start(context: Context) {
        synchronized(lock) {
            if (uplink != null) return
            uplink = OrionUplinkClient(context.applicationContext) { listener }.also { it.start() }
        }
    }

    fun stop() {
        val old = synchronized(lock) {
            val previous = uplink
            uplink = null
            previous
        }
        old?.stop()
    }

    /** Drops offline, oversized, malformed, or backpressured frames. */
    fun sendFrame(jpeg: ByteArray): Boolean = uplink?.sendFrame(jpeg) ?: false

    /** Avoid compressing screen images while no encrypted uplink is ready. */
    fun canSendFrames(): Boolean = uplink?.canSendFrames() ?: false

    /** For outbound JSON-RPC telemetry such as a user-approved UI snapshot. */
    fun sendJson(message: JSONObject): Boolean = uplink?.sendJson(message) ?: false

    /** Call when projection consent ends so the backend drops its last frame. */
    fun clearViewport(): Boolean = sendJson(JSONObject()
        .put("jsonrpc", "2.0").put("method", "viewport.stop")
        .put("params", JSONObject()))

    /** Reply only after the local UI approved and the device completed work. */
    fun sendCommandResult(id: String, result: JSONObject): Boolean =
        uplink?.sendCommandResult(id, result) ?: false

    /** Check immediately before executing a locally approved remote proposal. */
    fun isCommandPending(id: String): Boolean = uplink?.isCommandPending(id) ?: false

    /** Report denial or failure without performing the proposed action. */
    fun sendCommandError(id: String, code: Int, message: String): Boolean =
        uplink?.sendCommandError(id, code, message) ?: false
}

private class OrionUplinkClient(
    private val context: Context,
    private val listener: () -> OrionFrameTransport.Listener?,
) {
    companion object {
        private const val MAX_JPEG_BYTES = 1_000_000
        private const val MAX_QUEUED_BYTES = 1_500_000L
        private const val MAX_JSON_CHARS = 131_072
        private const val MAX_BACKOFF_MS = 30_000L
        private val JSON = "application/json; charset=utf-8".toMediaType()
    }

    private val worker = Executors.newSingleThreadScheduledExecutor { task ->
        Thread(task, "orion-mobile-uplink").apply { isDaemon = true }
    }
    private val http = pinnedHttpClient(context)
    @Volatile private var stopped = false
    @Volatile private var socket: WebSocket? = null
    @Volatile private var pendingSocket: WebSocket? = null
    private var tokenCall: Call? = null
    private var retry: ScheduledFuture<*>? = null
    private var ping: ScheduledFuture<*>? = null
    private var endpointIndex = 0
    private var failedRounds = 0
    private var pingSequence = 0L
    private var socketGeneration = 0L
    private val pendingCommands = ConcurrentHashMap<String, Long>()
    private val seenCommandIds = ConcurrentHashMap.newKeySet<String>()
    private val commandMethods = PhoneCommandContract.methods

    fun start() = dispatch { connect() }

    fun stop() {
        stopped = true
        retry?.cancel(false)
        ping?.cancel(false)
        val oldCall = tokenCall
        val oldSocket = socket
        val openingSocket = pendingSocket
        socket = null
        pendingSocket = null
        pendingCommands.clear()
        seenCommandIds.clear()
        worker.shutdownNow()
        closeHttpResources(http) {
            oldCall?.cancel()
            oldSocket?.cancel()
            if (openingSocket !== oldSocket) openingSocket?.cancel()
        }
        listener()?.onConnectionChanged(false, null)
    }

    fun sendFrame(jpeg: ByteArray): Boolean {
        if (stopped || jpeg.size !in 4..MAX_JPEG_BYTES ||
            jpeg[0] != 0xff.toByte() || jpeg[1] != 0xd8.toByte() ||
            jpeg[jpeg.lastIndex - 1] != 0xff.toByte() || jpeg[jpeg.lastIndex] != 0xd9.toByte()
        ) return false
        val ws = socket ?: return false
        if (ws.queueSize() + jpeg.size > MAX_QUEUED_BYTES) return false
        return ws.send(jpeg.toByteString())
    }

    fun canSendFrames(): Boolean = !stopped && socket != null

    fun sendJson(message: JSONObject): Boolean {
        if (stopped) return false
        val body = message.toString()
        if (body.length > MAX_JSON_CHARS) return false
        val ws = socket ?: return false
        if (ws.queueSize() + body.length > MAX_QUEUED_BYTES) return false
        return ws.send(body)
    }

    fun sendCommandResult(id: String, result: JSONObject): Boolean {
        // A successful result represents a real local approval and execution;
        // denied or failed proposals use sendCommandError instead.
        if (!result.optBoolean("authorized", false)) return false
        val expiresAt = pendingCommands.remove(id) ?: return false
        if (SystemClock.elapsedRealtime() > expiresAt) return false
        return sendJson(JSONObject().put("jsonrpc", "2.0").put("id", id)
            .put("result", result))
    }

    fun isCommandPending(id: String): Boolean {
        if (stopped || socket == null) return false
        val expiresAt = pendingCommands[id] ?: return false
        if (SystemClock.elapsedRealtime() <= expiresAt) return true
        pendingCommands.remove(id, expiresAt)
        return false
    }

    fun sendCommandError(id: String, code: Int, message: String): Boolean {
        if (pendingCommands.remove(id) == null) return false
        return sendJson(JSONObject().put("jsonrpc", "2.0").put("id", id)
            .put("error", JSONObject().put("code", code)
                .put("message", message.take(200))))
    }

    private fun dispatch(block: () -> Unit) {
        if (stopped) return
        try { worker.execute { if (!stopped) block() } }
        catch (_: java.util.concurrent.RejectedExecutionException) { }
    }

    private fun secureEndpoints(): List<HttpUrl> = Prefs.candidateUrls(context)
        .mapNotNull { it.toHttpUrlOrNull() }
        .filter { it.scheme == "https" && it.username.isEmpty() && it.password.isEmpty() }
        .distinctBy { "${it.host}:${it.port}" }

    private fun connect() {
        val credentials = Prefs.credentials(context)
        if (credentials == null) {
            listener()?.onPairingRequired()
            scheduleRetry(15_000)
            return
        }
        val endpoints = secureEndpoints()
        if (endpoints.isEmpty()) {
            // Never put a refresh token or a screen image on cleartext HTTP.
            listener()?.onConnectionChanged(false, null)
            scheduleRetry(MAX_BACKOFF_MS)
            return
        }
        if (endpointIndex >= endpoints.size) endpointIndex = 0
        val base = endpoints[endpointIndex]
        val tokenUrl = base.newBuilder().encodedPath("/v1/auth/token").query(null).build()
        val body = JSONObject()
            .put("device_id", credentials.first)
            .put("refresh_token", credentials.second)
            .toString().toRequestBody(JSON)
        val request = Request.Builder().url(tokenUrl).post(body).build()
        tokenCall = http.newCall(request).also { call ->
            call.enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) = dispatch {
                    if (tokenCall === call) tokenCall = null
                    nextEndpoint()
                }

                override fun onResponse(call: Call, response: Response) {
                    val code = response.code
                    val payload = response.use {
                        if (code != 200) null else runCatching {
                            val source = it.body?.source() ?: return@runCatching null
                            if (source.request(65_537) || source.buffer.size > 65_536)
                                return@runCatching null
                            JSONObject(source.readUtf8())
                        }.getOrNull()
                    }
                    dispatch {
                        tokenCall = null
                        when {
                            code == 401 || code == 403 -> {
                                listener()?.onPairingRequired()
                                scheduleRetry(60_000)
                            }
                            code == 429 -> scheduleRetry(30_000)
                            payload == null -> nextEndpoint()
                            else -> {
                                val token = payload.optString("access_token")
                                if (token.isEmpty()) nextEndpoint()
                                else {
                                    payload.optJSONArray("endpoints")?.let {
                                        Prefs.saveEndpoints(context, it.toString())
                                    }
                                    openSocket(base, token)
                                }
                            }
                        }
                    }
                }
            })
        }
    }

    private fun openSocket(base: HttpUrl, accessToken: String) {
        // OkHttp upgrades an HTTPS request to WSS. Bearer is carried only in
        // the TLS-protected upgrade request, never in the URL or logs.
        val url = base.newBuilder().encodedPath("/v1/device/uplink").query(null).build()
        val request = Request.Builder().url(url)
            .header("Authorization", "Bearer $accessToken").build()
        val generation = ++socketGeneration
        val ws = http.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                if (stopped) { webSocket.cancel(); return }
                dispatch {
                if (generation != socketGeneration) {
                    webSocket.cancel()
                    return@dispatch
                }
                if (pendingSocket === webSocket) pendingSocket = null
                socket = webSocket
                failedRounds = 0
                listener()?.onConnectionChanged(true, null)
                startPing()
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) = dispatch {
                if (generation == socketGeneration) handleText(text)
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(code, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) =
                dispatch { disconnected(webSocket, generation) }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) =
                dispatch { disconnected(webSocket, generation) }
        })
        // The listener may have opened before newWebSocket returns; onOpen owns
        // the authoritative socket reference.
        pendingSocket = ws
        if (stopped) ws.cancel()
    }

    private fun handleText(text: String) {
        if (text.length > MAX_JSON_CHARS) return
        val message = runCatching { JSONObject(text) }.getOrNull() ?: return
        val method = message.optString("method")
        if (method.isNotEmpty()) {
            val commandId = message.optString("id")
            if (message.optString("jsonrpc") != "2.0" ||
                method !in commandMethods || commandId.length !in 1..80 ||
                !message.optBoolean("requires_local_approval", false)
            ) {
                if (commandId.isNotEmpty()) {
                    sendJson(JSONObject().put("jsonrpc", "2.0")
                        .put("id", commandId)
                        .put("error", JSONObject().put("code", -32601)
                            .put("message", "Command not allowed")))
                }
                return
            }
            // Never reuse an id within this connection. A stale approval
            // dialog could otherwise authorise a different proposal sent
            // after the original request expires.
            if (commandId in seenCommandIds) return
            if (seenCommandIds.size >= 4096) {
                sendJson(JSONObject().put("jsonrpc", "2.0").put("id", commandId)
                    .put("error", JSONObject().put("code", -32000)
                        .put("message", "Command limit reached; reconnect the phone")))
                return
            }
            if (!seenCommandIds.add(commandId)) return
            pendingCommands[commandId] = SystemClock.elapsedRealtime() + 120_000
            val recipient = listener()
            if (recipient == null) {
                sendCommandError(commandId, -32000, "No local authorization UI")
            } else {
                // The listener receives a proposal only. It must never execute
                // directly from this network callback.
                try { recipient.onCommandRequest(message) }
                catch (_: Exception) {
                    sendCommandError(commandId, -32000,
                        "Local authorization UI unavailable")
                }
            }
            return
        }
        val id = message.optString("id")
        if (id.startsWith("ping-")) {
            val sentAt = id.removePrefix("ping-").toLongOrNull()
            if (sentAt != null) {
                listener()?.onConnectionChanged(true,
                    (SystemClock.elapsedRealtime() - sentAt).coerceAtLeast(0))
            }
        }
        listener()?.onServerMessage(message)
    }

    private fun disconnected(ws: WebSocket, generation: Long) {
        if (generation != socketGeneration) return
        socketGeneration++ // Ignore any later callback from this socket.
        if (socket === ws) socket = null
        if (pendingSocket === ws) pendingSocket = null
        pendingCommands.clear()
        seenCommandIds.clear()
        ping?.cancel(false)
        listener()?.onConnectionChanged(false, null)
        nextEndpoint()
    }

    private fun startPing() {
        ping?.cancel(false)
        ping = worker.scheduleWithFixedDelay({
            val id = "ping-${SystemClock.elapsedRealtime()}"
            pingSequence++
            sendJson(JSONObject().put("jsonrpc", "2.0").put("id", id)
                .put("method", "viewport.ping")
                .put("params", JSONObject().put("sequence", pingSequence)))
        }, 1, 15, TimeUnit.SECONDS)
    }

    private fun nextEndpoint() {
        if (stopped) return
        val count = secureEndpoints().size
        if (count == 0) {
            scheduleRetry(MAX_BACKOFF_MS)
        } else if (endpointIndex + 1 < count) {
            endpointIndex++
            scheduleRetry(0)
        } else {
            endpointIndex = 0
            failedRounds = min(failedRounds + 1, 5)
            val ceiling = min(MAX_BACKOFF_MS, 1_000L shl failedRounds)
            scheduleRetry(Random.nextLong(ceiling / 2, ceiling + 1))
        }
    }

    private fun scheduleRetry(delayMs: Long) {
        retry?.cancel(false)
        retry = worker.schedule({ if (!stopped) connect() }, delayMs, TimeUnit.MILLISECONDS)
    }
}

/** Closing TLS can write close_notify bytes. Always do this off the UI thread. */
internal fun closeHttpResources(client: OkHttpClient, cancel: () -> Unit = {}) {
    Thread({
        try { cancel() } catch (_: Exception) { }
        try {
            client.dispatcher.cancelAll()
            client.connectionPool.evictAll()
        } catch (_: Exception) { }
        finally { client.dispatcher.executorService.shutdown() }
    }, "orion-network-cleanup").apply { isDaemon = true }.start()
}

/** Android's system CA and host-name checks for every named HTTPS host, and
 *  exact certificate pinning for ORION's home Wi-Fi link (LanTrust). */
internal fun pinnedHttpClient(context: Context): OkHttpClient =
    LanTrust.secure(OkHttpClient.Builder(), context)
        .connectTimeout(5, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        // Authentication POSTs must never follow a redirect to another host.
        .followRedirects(false)
        .followSslRedirects(false)
        .build()
