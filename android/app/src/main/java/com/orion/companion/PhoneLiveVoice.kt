package com.orion.companion

import android.content.Context
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.io.Closeable
import java.net.URLEncoder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * A warm Live speech session. PhoneBrain remains the sole source of replies and
 * actions. Live audio is generative, so spoken wording may differ slightly from
 * the approved answer retained in the phone terminal.
 */
internal class PhoneLiveVoice(context: Context) : Closeable {
    private val app = context.applicationContext
    private val gate = Any()
    private val http = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS)
        .pingInterval(15, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false)
        .followRedirects(false).followSslRedirects(false).build()
    private var socket: WebSocket? = null
    private var ready: CountDownLatch? = null
    private var inbox = ArrayBlockingQueue<PhoneLiveProtocol.Event>(8)
    private var openedAt = 0L
    private var connectionVoice = ""
    private var pauseUntil = 0L
    private var connectionId = 0L
    private var closed = false

    fun prewarm() {
        synchronized(gate) {
            if (closed || System.currentTimeMillis() < pauseUntil) return
            val voice = PhoneVoices.cloudName(PhoneAiProfile.voiceName(app))
            if (socket != null && connectionVoice == voice &&
                System.currentTimeMillis() - openedAt < TimeUnit.MINUTES.toMillis(8)) return
            resetLocked()
            val id = ++connectionId
            val key = PhoneAiProfile.key(app) ?: return
            val wait = CountDownLatch(1)
            val events = ArrayBlockingQueue<PhoneLiveProtocol.Event>(8)
            ready = wait
            inbox = events
            openedAt = System.currentTimeMillis()
            connectionVoice = voice
            val url = "wss://generativelanguage.googleapis.com/ws/" +
                "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=" +
                URLEncoder.encode(key, "UTF-8")
            val request = Request.Builder().url(url).build()
            val ws = http.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    if (!webSocket.send(PhoneLiveProtocol.setup(voice))) fail(id, webSocket, 30_000L)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    val event = runCatching { PhoneLiveProtocol.parse(text) }.getOrNull()
                    if (event == null || event.error) { fail(id, webSocket, 30_000L); return }
                    synchronized(gate) {
                        if (connectionId != id) return
                        if (event.ready) wait.countDown()
                        if ((!event.ready || event.audio.isNotEmpty()) && !events.offer(event)) {
                            fail(id, webSocket, 30_000L)
                        }
                    }
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    fail(id, webSocket, LiveReconnectPolicy.pauseMillis(response?.code))
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    fail(id, webSocket, 30_000L)
                }
            })
            if (connectionId == id) socket = ws else ws.cancel()
        }
    }

    /** Return PARTIAL once any audio played; replaying a spoken answer is unsafe. */
    fun stream(reply: String, cancelled: () -> Boolean, play: (ByteArray) -> Unit): LiveSpeechResult {
        prewarm()
        val wait: CountDownLatch
        val events: ArrayBlockingQueue<PhoneLiveProtocol.Event>
        val ws: WebSocket
        synchronized(gate) {
            wait = ready ?: return LiveSpeechResult.UNAVAILABLE
            events = inbox
            ws = socket ?: return LiveSpeechResult.UNAVAILABLE
        }
        val guard = LiveTurnGuard()
        try {
            if (!wait.await(2, TimeUnit.SECONDS) || cancelled())
                return guard.failed(cancelled())
            synchronized(gate) {
                if (socket !== ws || !ws.send(PhoneLiveProtocol.turn(reply)))
                    return LiveSpeechResult.UNAVAILABLE
            }
            val deadline = System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(3)
            while (!cancelled() && System.currentTimeMillis() < deadline) {
                val event = events.poll(if (guard.audioMayHavePlayed) 25 else 12, TimeUnit.SECONDS)
                    ?: return guard.failed()
                if (event.error || event.interrupted || event.goAway)
                    return guard.failed()
                for (pcm in event.audio) {
                    if (cancelled()) return LiveSpeechResult.CANCELLED
                    // Mark the turn before writing: AudioTrack can fail after
                    // playing only part of a chunk, which must never be replayed.
                    guard.beforePlayback()
                    play(pcm)
                }
                if (event.complete) return guard.completed()
            }
            return guard.failed(cancelled())
        } catch (_: Exception) {
            return guard.failed(cancelled())
        } finally {
            // A broken or interrupted turn has no trustworthy position in the
            // Live session; the next reply starts on a new connection.
            if (!guard.finishedSuccessfully || cancelled()) reset()
        }
    }

    private fun fail(id: Long, ws: WebSocket, pause: Long) {
        synchronized(gate) {
            if (connectionId != id) return
            connectionId++
            pauseUntil = System.currentTimeMillis() + pause
            inbox.clear()
            inbox.offer(PhoneLiveProtocol.Event(error = true))
            ready?.countDown()
            socket = null
        }
        ws.cancel()
    }

    fun reset() = synchronized(gate) { resetLocked() }

    private fun resetLocked() {
        connectionId++
        val old = socket
        socket = null
        inbox.clear()
        inbox.offer(PhoneLiveProtocol.Event(error = true))
        ready?.countDown()
        ready = null
        old?.cancel()
    }

    override fun close() {
        synchronized(gate) { closed = true; resetLocked() }
        closeHttpResources(http)
    }
}
