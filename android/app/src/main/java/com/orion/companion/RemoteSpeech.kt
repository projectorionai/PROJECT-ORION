package com.orion.companion

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.PlaybackParams
import android.os.Handler
import android.os.Looper
import okhttp3.Call
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Host-generated cloud audio, with an authorised phone voice if it is unavailable. */
class RemoteSpeech(
    context: Context,
    private val onSpeaking: (Boolean) -> Unit = {},
    private val onVoice: (String) -> Unit = {},
) {
    companion object {
        private val speakers = java.util.concurrent.atomic.AtomicInteger()
        private val instances = java.util.WeakHashMap<RemoteSpeech, Boolean>()
        fun anySpeaking(): Boolean = speakers.get() > 0
        fun stopAll(expectedPlayback: Long? = null) {
            val action = {
                val live = synchronized(instances) { instances.keys.toList() }
                live.forEach { it.stop(expectedPlayback) }
            }
            if (Looper.myLooper() == Looper.getMainLooper()) action()
            else Handler(Looper.getMainLooper()).post(action)
        }
    }
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val turnGate = Any()
    private val generation = AtomicLong()
    private var playbackEpoch = 0L
    private val worker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-charon-speech").apply { isDaemon = true }
    }
    private val deviceSpeech = PhoneDeviceSpeech(app)
    private val client = pinnedHttpClient(app).newBuilder()
        .readTimeout(50, TimeUnit.SECONDS).callTimeout(55, TimeUnit.SECONDS).build()
    @Volatile private var closed = false
    @Volatile private var call: Call? = null
    @Volatile var isSpeaking = false
        private set
    @Volatile private var streamPlayer: AudioTrack? = null
    private var tokenEndpoint = ""
    private var tokenCredentials: Pair<String, String>? = null
    private var tokenValue = ""
    private var tokenExpires = 0L
    private enum class StreamResult { COMPLETE, UNAVAILABLE, INTERRUPTED, UNSUPPORTED }
    init { synchronized(instances) { instances[this] = true } }

    private fun state(active: Boolean) {
        if (active != isSpeaking) {
            if (active) speakers.incrementAndGet() else speakers.decrementAndGet()
        }
        isSpeaking = active
        onSpeaking(active)
    }

    fun speak(text: String) {
        val sections = SpeechText.parts(text)
        if (sections.isEmpty()) return
        val scheduled = synchronized(turnGate) {
            if (closed) return
            playbackEpoch = SpeechPlaybackEpoch.begin()
            generation.incrementAndGet()
        }
        main.post {
            val id = synchronized(turnGate) {
                if (closed || generation.get() != scheduled) return@post
                stop()
                generation.get()
            }
            // Keep the microphone suppressed between all sections as well
            // as during playback, so the phone cannot transcribe its own reply.
            state(true)
            worker.execute {
              var voiceStatus: String? = null
              val output = ReplyAudio(id)
              var phoneOnly = PhoneAiProfile.voiceName(app) == PhoneVoices.PHONE
              var requests = 0
              fun phoneSpeech(section: String): Boolean {
                  output.finish()
                  if (closed || id != generation.get()) return false
                  PhonePerformance.voice(app, "Phone", "android-tts")
                  return deviceSpeech.speak(section, { closed || id != generation.get() })
              }
              try {
                for (spoken in sections) {
                    if (closed || id != generation.get()) return@execute
                    if (phoneOnly) {
                        if (!phoneSpeech(spoken)) break
                        continue
                    }
                    requests++
                    val streamed = streamAudio(spoken, id, output)
                    if (closed || id != generation.get()) return@execute
                    if (streamed == StreamResult.COMPLETE) continue
                    phoneOnly = true
                    PhonePerformance.voiceFallback(app, if (streamed == StreamResult.INTERRUPTED)
                        "Partial audio" else "Unavailable")
                    if (streamed == StreamResult.INTERRUPTED) {
                        // Drain the partial section; only later untouched text
                        // can be read safely by the phone engine.
                        output.finish()
                        voiceStatus = "Voice interrupted. The full reply is shown below."
                    } else if (!phoneSpeech(spoken)) {
                        voiceStatus = "Voice unavailable. The full reply is shown below."
                        break
                    }
                }
                output.finish()
                PhonePerformance.speechDelivery(app, requests, output.removedSamples)
              } catch (_: Exception) {
                voiceStatus = "Voice interrupted. The full reply is shown below."
              } finally {
                output.close()
                main.post { if (!closed && id == generation.get()) {
                    state(false)
                    voiceStatus?.let(onVoice)
                } }
              }
            }
        }
    }

    private fun accessToken(endpoint: String, id: Long, renew: Boolean = false): String? {
        val credentials = Prefs.credentials(app) ?: return null
        val now = System.nanoTime()
        if (!renew && endpoint == tokenEndpoint && credentials == tokenCredentials &&
            now < tokenExpires && tokenValue.isNotBlank()) return tokenValue
        val base = endpoint.toHttpUrlOrNull() ?: return null
        if (base.scheme != "https") return null
        val body = JSONObject().put("device_id", credentials.first)
            .put("refresh_token", credentials.second).toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType())
        val request = Request.Builder().url(base.newBuilder().encodedPath("/v1/auth/token")
            .query(null).build()).post(body).build()
        val token = withResponse(request, id) { response ->
            if (!response.isSuccessful) null else {
                val data = JSONObject(response.body?.string().orEmpty())
                val value = data.optString("access_token")
                if (value.isBlank()) null else {
                    tokenEndpoint = endpoint
                    tokenCredentials = credentials
                    tokenValue = value
                    // Access tokens last fifteen minutes. Keep them in memory
                    // and renew early; no extra round trip for every sentence.
                    val ttl = data.optLong("expires_in", 900).coerceIn(1, 900)
                    tokenExpires = now + (ttl - minOf(60, ttl / 2)) * 1_000_000_000L
                    value
                }
            }
        }
        return token
    }

    private inner class ReplyAudio(private val id: Long) : java.io.Closeable {
        private var track: AudioTrack? = null
        private var writtenFrames = 0L
        private val pauses = PcmPauseFilter()
        private var finished = false
        val removedSamples: Long get() = pauses.removedSamples
        private fun cancelled() = closed || id != generation.get() || Thread.currentThread().isInterrupted
        private val output = PcmReplyPlayback(::cancelled, { pcm ->
            if (track == null) {
                val minimum = AudioTrack.getMinBufferSize(24_000,
                    AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
                check(minimum > 0)
                val opened = AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
                    .setAudioFormat(AudioFormat.Builder().setSampleRate(24_000)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
                    .setBufferSizeInBytes(maxOf(minimum * 2, 9_600))
                    .setTransferMode(AudioTrack.MODE_STREAM).build()
                track = opened
                check(opened.state == AudioTrack.STATE_INITIALIZED)
                runCatching { opened.playbackParams = PlaybackParams().setSpeed(1.10f).setPitch(1f) }
                synchronized(turnGate) {
                    check(!cancelled()) { "Speech stopped" }
                    streamPlayer = opened
                    opened.play()
                }
            }
            val opened = track!!
            var offset = 0
            while (offset < pcm.size && !cancelled()) {
                val count = opened.write(pcm, offset, minOf(4096, pcm.size - offset), AudioTrack.WRITE_BLOCKING)
                check(count > 0) { "Speech playback stopped" }
                writtenFrames += count / 2L
                offset += count
            }
            check(offset == pcm.size) { "Speech stopped" }
        }, {
            val opened = track
            if (opened != null) {
                val remaining = (writtenFrames - opened.playbackHeadPosition.toLong()).coerceAtLeast(0L)
                val deadline = System.nanoTime() + (remaining * 1_000_000_000L / 24_000L) + 3_000_000_000L
                while (!cancelled() && opened.playbackHeadPosition.toLong() < writtenFrames &&
                    System.nanoTime() < deadline) Thread.sleep(15)
                check(cancelled() || opened.playbackHeadPosition.toLong() >= writtenFrames) {
                    "Speech playback was truncated"
                }
            }
        }, {
            val opened = track
            synchronized(turnGate) { if (streamPlayer === opened) streamPlayer = null }
            runCatching { opened?.stop() }
            runCatching { opened?.release() }
        })

        fun append(pcm: ByteArray) {
            val filtered = pauses.add(pcm)
            if (filtered.isNotEmpty()) output.append(filtered)
        }
        fun finish() {
            if (finished) return
            finished = true
            val tail = pauses.finish()
            if (tail.isNotEmpty()) output.append(tail)
            output.finish()
        }
        override fun close() = output.close()
    }

    private fun streamAudio(text: String, id: Long, output: ReplyAudio): StreamResult {
        val voice = PhoneVoices.cloudName(PhoneAiProfile.voiceName(app))
        val endpoints = (listOfNotNull(Prefs.baseUrl(app)) + Prefs.candidateUrls(app)).distinct()
        for (endpoint in endpoints) {
            if (closed || id != generation.get()) return StreamResult.INTERRUPTED
            val base = endpoint.toHttpUrlOrNull() ?: continue
            if (base.scheme != "https") continue
            var token = runCatching { accessToken(endpoint, id) }.getOrNull() ?: continue
            // A 401 is rejected before provider generation, so one token
            // renewal is safe. Other failures must never replay paid speech.
            for (attempt in 0..1) {
                var renew = false
                val request = Request.Builder().url(base.newBuilder()
                    .encodedPath("/api/speech/stream").query(null).build())
                    .header("Authorization", "Bearer $token")
                    .post(JSONObject().put("text", text).put("voice", voice).toString()
                        .toRequestBody("application/json; charset=utf-8".toMediaType())).build()
                var received = false
                val result = runCatching {
                    withResponse(request, id, streaming = true) { response ->
                        if (response.code == 401 && attempt == 0) {
                            renew = true
                            return@withResponse StreamResult.UNAVAILABLE
                        }
                        if (response.code in listOf(404, 405)) return@withResponse StreamResult.UNSUPPORTED
                        if (!response.isSuccessful || response.header("X-Orion-Voice") != voice ||
                            response.header("X-Orion-PCM-Format") != "s16le-mono-v1" ||
                            response.header("X-Orion-Sample-Rate") != "24000")
                            return@withResponse StreamResult.UNAVAILABLE
                        val input = response.body?.byteStream() ?: return@withResponse StreamResult.UNAVAILABLE
                        val frames = PcmFrames(input)
                        while (!closed && id == generation.get()) {
                            val pcm = frames.next() ?: break
                            if (closed || id != generation.get()) return@withResponse StreamResult.INTERRUPTED
                            received = true // Pending audio can already be playing on its worker.
                            output.append(pcm)
                        }
                        if (closed || id != generation.get()) return@withResponse StreamResult.INTERRUPTED
                        check(input.read() == -1) { "Audio arrived after speech completion" }
                        if (received) StreamResult.COMPLETE else StreamResult.UNAVAILABLE
                    } ?: StreamResult.UNAVAILABLE
                }.getOrElse {
                    if (received) StreamResult.INTERRUPTED else StreamResult.UNAVAILABLE
                }
                if (!renew) {
                    // No audio was played for these outcomes. Another saved
                    // HTTPS route may still reach the same PC after a handoff.
                    if (result == StreamResult.UNAVAILABLE || result == StreamResult.UNSUPPORTED) break
                    return result
                }
                token = runCatching { accessToken(endpoint, id, renew = true) }.getOrNull()
                    ?: break
            }
        }
        return StreamResult.UNAVAILABLE
    }

    private fun <T> withResponse(request: Request, id: Long, streaming: Boolean = false,
                                 consume: (okhttp3.Response) -> T): T? {
        if (closed || id != generation.get()) return null
        val current = client.newCall(request)
        if (streaming) current.timeout().timeout(210, TimeUnit.SECONDS)
        call = current
        try {
            if (closed || id != generation.get()) { current.cancel(); return null }
            return current.execute().use(consume)
        } finally { if (call === current) call = null }
    }

    fun stop(expectedPlayback: Long? = null) {
        val (previous, stream) = synchronized(turnGate) {
            if (expectedPlayback != null && playbackEpoch != expectedPlayback) return
            generation.incrementAndGet()
            val previous = call
            val stream = streamPlayer
            streamPlayer = null
            previous to stream
        }
        Thread({ previous?.cancel() }, "orion-speech-cancel").apply { isDaemon = true }.start()
        deviceSpeech.stop()
        runCatching { stream?.pause(); stream?.flush(); stream?.stop() }
        state(false)
    }

    fun close() {
        closed = true
        synchronized(instances) { instances.remove(this) }
        stop()
        worker.shutdownNow()
        deviceSpeech.close()
        closeHttpResources(client)
    }
}
