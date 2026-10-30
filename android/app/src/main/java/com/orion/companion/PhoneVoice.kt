package com.orion.companion

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.media.PlaybackParams
import android.os.Handler
import android.os.Looper
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.security.MessageDigest
import java.util.Base64
import java.util.WeakHashMap
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

internal object PhoneVoiceCooldown {
    fun delayMillis(code: Int, detail: String, retryAfterSeconds: Long?, now: Long): Long = when (code) {
        400, 403, 404 -> TimeUnit.HOURS.toMillis(24)
        429 -> if (PhoneTextModels.dailyQuota(detail))
            PhoneTextModels.pauseMillis(code, retryAfterSeconds, daily = true, nowEpochMillis = now)
        else maxOf(TimeUnit.MINUTES.toMillis(3),
            (retryAfterSeconds ?: 0L).coerceIn(0L, 900L) * 1_000L)
        500, 502, 503, 504 -> TimeUnit.MINUTES.toMillis(2)
        else -> TimeUnit.MINUTES.toMillis(1)
    }
}

/** Direct cloud speech, with an authorised phone voice when cloud audio is unavailable. */
class PhoneVoice(
    context: Context,
    private val onSpeaking: (Boolean) -> Unit = {},
    private val onStatus: (String) -> Unit = {},
) {
    companion object {
        private val speakers = AtomicInteger()
        private val instances = WeakHashMap<PhoneVoice, Boolean>()
        fun anySpeaking(): Boolean = speakers.get() > 0
        @Volatile var outputLevel = 0f
            private set
        fun stopAll(expectedPlayback: Long? = null) {
            val action = {
                synchronized(instances) { instances.keys.toList() }.forEach { it.stop(expectedPlayback) }
            }
            if (Looper.myLooper() == Looper.getMainLooper()) action()
            else Handler(Looper.getMainLooper()).post(action)
        }
        /** Starting a normal turn should preserve an idle, prepared cloud connection. */
        fun stopActiveAll() {
            val action = {
                synchronized(instances) { instances.keys.toList() }.forEach { voice ->
                    synchronized(voice.turnGate) {
                        if (voice.isSpeaking || voice.incrementalTurn != null || voice.speechQueued) voice.stop()
                    }
                }
            }
            if (Looper.myLooper() == Looper.getMainLooper()) action()
            else Handler(Looper.getMainLooper()).post(action)
        }
        private const val RATE = 24_000
        private const val MAX_AUDIO_BYTES = RATE * 2 * 180
        private val unavailableUntil = ConcurrentHashMap<String, Long>()
        private fun availabilityKey(model: String, credential: String): String {
            val digest = MessageDigest.getInstance("SHA-256")
                .digest(credential.toByteArray(Charsets.UTF_8))
            val account = digest.take(8).joinToString("") { "%02x".format(it.toInt() and 0xff) }
            return "$account:$model"
        }
    }

    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val turnGate = Any()
    private val generation = AtomicLong()
    private val firstAudioStartedAt = AtomicLong()
    private val worker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-phone-charon").apply { isDaemon = true }
    }
    private val client = OkHttpClient.Builder()
        .connectTimeout(8, TimeUnit.SECONDS).readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(210, TimeUnit.SECONDS).retryOnConnectionFailure(false)
        .followRedirects(false).followSslRedirects(false).build()
    private val live = PhoneLiveVoice(app)
    private val deviceSpeech = PhoneDeviceSpeech(app)
    @Volatile private var incrementalTurn: IncrementalTurn? = null
    @Volatile private var closed = false
    @Volatile private var call: Call? = null
    @Volatile private var player: AudioTrack? = null
    private var playbackEpoch = 0L
    private var speechQueued = false
    @Volatile var isSpeaking = false
        private set
    private enum class Result { COMPLETE, UNAVAILABLE, CONNECTION_ERROR, PARTIAL, CANCELLED }

    init {
        synchronized(instances) { instances[this] = true }
        // Open the Charon session while transcription and phone actions run.
        if (PhoneAiProfile.voiceName(app) != PhoneVoices.PHONE && PhoneAiProfile.liveVoiceEnabled(app))
            worker.execute { live.prewarm() }
    }

    /** Refresh the idle Charon connection whilst transcription or text generation runs. */
    fun prepare() {
        if (!closed && PhoneAiProfile.voiceName(app) != PhoneVoices.PHONE && PhoneAiProfile.liveVoiceEnabled(app))
            worker.execute { if (!closed) live.prewarm() }
    }

    private fun state(active: Boolean) {
        if (active != isSpeaking) {
            if (active) speakers.incrementAndGet() else speakers.decrementAndGet()
            isSpeaking = active
            if (!active) outputLevel = 0f
            onSpeaking(active)
        }
    }

    fun speak(text: String) {
        val sections = SpeechText.parts(text)
        if (sections.isEmpty()) return
        val id = synchronized(turnGate) {
            if (closed) return
            if (isSpeaking || incrementalTurn != null || speechQueued) stop()
            val scheduled = generation.incrementAndGet()
            playbackEpoch = SpeechPlaybackEpoch.begin()
            speechQueued = true
            firstAudioStartedAt.set(PhonePerformance.now())
            scheduled
        }
        main.post {
            synchronized(turnGate) {
                if (cancelled(id)) return@post
                speechQueued = false
                state(true) // Keep the microphone suppressed between every queued section.
            }
            worker.execute {
                val output = ReplyAudio(id)
                try {
                    val models = SpeechText.voiceModels(PhoneAiProfile.voiceModel(app))
                    for (section in sections) {
                        if (!playSection(section, id, models, output)) break
                    }
                    output.finish()
                } catch (_: Exception) {
                    if (!cancelled(id)) status(id, "Voice interrupted. The full reply is shown below.")
                } finally {
                    output.close()
                    main.post { if (!cancelled(id)) state(false) }
                }
            }
        }
    }

    /** A single approved reply. Enqueued sentences play in order without restarting Charon. */
    inner class IncrementalTurn internal constructor(private val id: Long) {
        private val pending = LinkedBlockingQueue<String>()
        @Volatile private var finished = false

        fun enqueue(text: String): Boolean {
            val sections = SpeechText.parts(text)
            if (sections.isEmpty()) return false
            synchronized(this) {
                if (finished || cancelled(id) || incrementalTurn !== this) return false
                sections.forEach { pending.add(it) }
                return true
            }
        }

        fun finish() { synchronized(this) { finished = true } }

        fun cancel() {
            synchronized(turnGate) { if (incrementalTurn === this) stop() }
        }

        internal fun next(): String? {
            while (!cancelled(id)) {
                pending.poll(50, TimeUnit.MILLISECONDS)?.let { first ->
                    // Text commonly arrives much faster than speech. Join the
                    // waiting sentences into one request whilst prior PCM plays.
                    // The first sentence still starts as soon as it is ready.
                    val section = StringBuilder(first)
                    while (true) {
                        val next = pending.peek() ?: break
                        if (section.length + next.length + 1 > 1200) break
                        pending.poll() ?: break
                        section.append(' ').append(next)
                    }
                    return section.toString()
                }
                if (finished && pending.isEmpty()) return null
            }
            return null
        }
    }

    /** Start after the first complete sentence is approved; finish after the last one. */
    fun beginIncremental(): IncrementalTurn {
        val (turn, id) = synchronized(turnGate) {
            if (closed) return IncrementalTurn(generation.get())
            if (isSpeaking || incrementalTurn != null || speechQueued) stop()
            val id = generation.incrementAndGet()
            playbackEpoch = SpeechPlaybackEpoch.begin()
            val turn = IncrementalTurn(id)
            incrementalTurn = turn
            firstAudioStartedAt.set(PhonePerformance.now())
            turn to id
        }
        main.post { if (!cancelled(id)) state(true) }
        worker.execute {
            val output = ReplyAudio(id)
            try {
                val models = SpeechText.voiceModels(PhoneAiProfile.voiceModel(app))
                while (!cancelled(id)) {
                    val section = turn.next() ?: break
                    if (!playSection(section, id, models, output)) break
                }
                output.finish()
            } catch (_: Exception) {
                if (!cancelled(id)) status(id, "Voice interrupted. The full reply is shown below.")
            } finally {
                output.close()
                synchronized(turnGate) { if (incrementalTurn === turn) incrementalTurn = null }
                main.post { if (!cancelled(id)) state(false) }
            }
        }
        return turn
    }

    private fun playSection(section: String, id: Long, models: List<String>, output: ReplyAudio): Boolean {
        if (cancelled(id)) return false
        if (output.phoneOnly) return output.speakPhone(section)
        output.requests++
        val credential = PhoneAiProfile.key(app).orEmpty()
        fun directResult(): Result =
            models.asSequence().filter { model ->
                (unavailableUntil[availabilityKey(model, credential)] ?: 0L) <= System.currentTimeMillis()
            }.map { model -> stream(section, model, id, output) }
                .firstOrNull { it != Result.UNAVAILABLE } ?: Result.UNAVAILABLE
        val preferLive = PhoneAiProfile.liveVoiceEnabled(app)
        val first = if (preferLive) streamLive(section, id, output) else directResult()
        // A failed route can be retried elsewhere only before any audio played.
        // PARTIAL must never be replayed because PCM has no word offset.
        val result = if (preferLive &&
            (first == Result.UNAVAILABLE || first == Result.CONNECTION_ERROR)) directResult() else first
        return when (result) {
            Result.COMPLETE -> true
            Result.CANCELLED -> false
            Result.CONNECTION_ERROR -> {
                output.usePhone("Unavailable")
                output.speakPhone(section)
            }
            Result.UNAVAILABLE -> {
                output.usePhone("Unavailable")
                output.speakPhone(section)
            }
            Result.PARTIAL -> {
                // The current section has no reliable word offset. Drain its
                // partial audio, then read only subsequent, untouched sections.
                output.usePhone("Partial audio")
                status(id, "Voice interrupted. The full reply is shown below.")
                true
            }
        }
    }

    private fun status(id: Long, message: String) {
        main.post { if (!cancelled(id)) onStatus(message) }
    }

    private fun cancelled(id: Long) = closed || generation.get() != id || Thread.currentThread().isInterrupted

    private fun firstAudio(id: Long, route: String, model: String) {
        if (cancelled(id)) return
        val started = firstAudioStartedAt.getAndSet(0L)
        if (started > 0L) {
            PhonePerformance.voice(app, route, model)
            PhonePerformance.record(app, "first_audio", started)
        }
    }

    /** Keep the device open and fetch the next section whilst prior audio plays. */
    private inner class ReplyAudio(private val id: Long) : java.io.Closeable {
        var phoneOnly = PhoneAiProfile.voiceName(app) == PhoneVoices.PHONE
            private set
        var requests = 0
        private var cloudFinished = false
        private val pauses = PcmPauseFilter()
        private var track: AudioTrack? = null
        private var writtenFrames = 0L
        private var route = ""
        private var model = ""
        private val output = PcmReplyPlayback({ cancelled(id) }, { pcm ->
            if (track == null) {
                val opened = createTrack()
                track = opened
                // stop() and renderer initialisation share the turn gate, so
                // a cancelled turn cannot publish a new live device afterwards.
                synchronized(turnGate) {
                    check(!cancelled(id)) { "Speech stopped" }
                    player = opened
                    opened.play()
                }
                firstAudio(id, route, model)
            }
            val opened = track!!
            var offset = 0
            while (offset < pcm.size && !cancelled(id)) {
                val count = opened.write(pcm, offset, minOf(4096, pcm.size - offset), AudioTrack.WRITE_BLOCKING)
                check(count > 0) { "Speech playback stopped" }
                outputLevel = PcmAudioLevel.of(pcm, offset, count)
                writtenFrames += count / 2L
                offset += count
            }
            check(offset == pcm.size) { "Speech stopped" }
        }, {
            val opened = track
            if (opened != null) {
                val remaining = (writtenFrames - opened.playbackHeadPosition.toLong()).coerceAtLeast(0L)
                val deadline = android.os.SystemClock.elapsedRealtime() + remaining * 1000L / RATE + 3_000L
                while (!cancelled(id) && opened.playbackHeadPosition.toLong() < writtenFrames &&
                    android.os.SystemClock.elapsedRealtime() < deadline) Thread.sleep(15)
                check(cancelled(id) || opened.playbackHeadPosition.toLong() >= writtenFrames) {
                    "Speech playback was truncated"
                }
            }
        }, {
            val opened = track
            synchronized(turnGate) { if (player === opened) player = null }
            runCatching { opened?.stop() }
            runCatching { opened?.release() }
        })

        fun append(pcm: ByteArray, route: String, model: String) {
            if (!output.hasAudio) { this.route = route; this.model = model }
            val filtered = pauses.add(pcm)
            if (filtered.isNotEmpty()) output.append(filtered)
        }
        fun usePhone(reason: String) {
            finishCloud()
            phoneOnly = true
            PhonePerformance.voiceFallback(app, reason)
        }
        fun speakPhone(text: String): Boolean {
            finishCloud()
            if (cancelled(id)) return false
            if (PhoneAiProfile.voiceName(app) == PhoneVoices.PHONE)
                PhonePerformance.voiceFallback(app, "Selected")
            PhonePerformance.voice(app, "Phone", "android-tts")
            val complete = deviceSpeech.speak(text, { cancelled(id) }) {
                firstAudio(id, "Phone", "android-tts")
            }
            if (!complete && !cancelled(id)) status(id, "Voice unavailable. The full reply is shown below.")
            return complete
        }
        private fun finishCloud() {
            if (cloudFinished) return
            cloudFinished = true
            val tail = pauses.finish()
            if (tail.isNotEmpty()) output.append(tail)
            output.finish()
        }
        fun finish() {
            finishCloud()
            PhonePerformance.speechDelivery(app, requests, pauses.removedSamples)
        }
        override fun close() = output.close()
    }

    private fun streamLive(text: String, id: Long, output: ReplyAudio): Result {
        var received = false
        var total = 0
        try {
            val result = live.stream(text, { cancelled(id) }) { pcm ->
                check(pcm.size % 2 == 0 && pcm.isNotEmpty())
                total += pcm.size
                check(total <= MAX_AUDIO_BYTES) { "Live speech is too long" }
                if (cancelled(id)) return@stream
                received = true // Queued audio can play even before the device starts.
                output.append(pcm, "Live", PhoneLiveProtocol.MODEL)
            }
            when (result) {
                LiveSpeechResult.UNAVAILABLE -> return Result.UNAVAILABLE
                LiveSpeechResult.PARTIAL -> return Result.PARTIAL
                LiveSpeechResult.CANCELLED -> return Result.CANCELLED
                LiveSpeechResult.COMPLETE -> Unit
            }
            if (cancelled(id)) return Result.CANCELLED
            return if (received) Result.COMPLETE else Result.UNAVAILABLE
        } catch (_: Exception) {
            live.reset()
            return if (cancelled(id)) Result.CANCELLED else if (received) Result.PARTIAL
                else Result.UNAVAILABLE
        }
    }

    private fun stream(text: String, model: String, id: Long, output: ReplyAudio): Result {
        val key = PhoneAiProfile.key(app) ?: return Result.UNAVAILABLE
        val modern = model.startsWith("gemini-3.8-")
        val unary = model == "gemini-2.5-flash-preview-tts"
        val style = "British English, composed and precise, brisk conversation, short natural pauses."
        val selectedVoice = PhoneVoices.cloudName(PhoneAiProfile.voiceName(app))
        val part = JSONObject().put("text", if (modern) text else
            "Read exactly this text in British English, clearly and briskly, with short natural pauses. " +
                "Do not add words:\n\n$text")
        if (modern) part.put("speech_metadata", JSONObject().put("style", style))
        val voiceConfig = if (modern) JSONObject().put("voice", selectedVoice) else
            JSONObject().put("prebuiltVoiceConfig", JSONObject().put("voiceName", selectedVoice))
        val config = JSONObject().put("responseModalities", JSONArray().put("AUDIO"))
            .put("speechConfig", JSONObject().put("voiceConfig", voiceConfig))
        if (modern) config.put("responseFormat", JSONObject().put("audio", JSONObject()
            .put("mimeType", "AUDIO_L16").put("sampleRate", RATE)))
        val payload = JSONObject().put("contents", JSONArray().put(JSONObject().put("role", "user")
            .put("parts", JSONArray().put(part)))).put("generationConfig", config)
        val request = Request.Builder()
            .url("https://generativelanguage.googleapis.com/v1beta/models/$model:" +
                if (unary) "generateContent" else "streamGenerateContent?alt=sse")
            .header("x-goog-api-key", key).header("Accept", if (unary) "application/json" else "text/event-stream")
            .post(payload.toString().toRequestBody("application/json; charset=utf-8".toMediaType())).build()
        val current = client.newCall(request)
        var received = false
        call = current
        try {
            if (cancelled(id)) { current.cancel(); return Result.CANCELLED }
            current.execute().use { response ->
                if (!response.isSuccessful) {
                    rememberFailure(model, key, response.code,
                        response.peekBody(16_384).string(), response.header("Retry-After")?.toLongOrNull())
                    return Result.UNAVAILABLE
                }
                val contentType = response.header("Content-Type").orEmpty()
                check(contentType.startsWith(if (unary) "application/json" else "text/event-stream"))
                val body = response.body ?: return Result.UNAVAILABLE
                val events = if (unary) null else PhoneSseEvents(body.byteStream())
                var unaryRead = false
                var total = 0
                var finished = false
                while (!cancelled(id)) {
                    val raw = if (unary) {
                        if (unaryRead) null else { unaryRead = true; body.string() }
                    } else events?.next()
                    if (raw == null) break
                    if (raw == "[DONE]") continue
                    val event = JSONObject(raw)
                    check(!event.has("error")) { "Speech generation failed" }
                    val candidate = event.optJSONArray("candidates")?.optJSONObject(0) ?: continue
                    val parts = candidate.optJSONObject("content")?.optJSONArray("parts")
                    if (parts != null) for (i in 0 until parts.length()) {
                        val inline = parts.optJSONObject(i)?.optJSONObject("inlineData") ?: continue
                        val encoded = inline.optString("data")
                        if (encoded.isEmpty()) continue
                        check(!finished) { "Audio arrived after completion" }
                        val mime = inline.optString("mimeType").lowercase()
                        check(mime.startsWith("audio/l16") || mime.startsWith("audio/pcm"))
                        val rate = Regex("(?:^|;)\\s*rate=(\\d+)").find(mime)?.groupValues?.get(1)?.toInt() ?: RATE
                        val channels = Regex("(?:^|;)\\s*channels=(\\d+)").find(mime)?.groupValues?.get(1)?.toInt() ?: 1
                        check(rate == RATE && channels == 1)
                        val pcm = Base64.getDecoder().decode(encoded)
                        check(pcm.isNotEmpty() && pcm.size % 2 == 0 && pcm.size <= 1_572_864)
                        total += pcm.size
                        check(total <= MAX_AUDIO_BYTES) { "Speech audio is too long" }
                        if (cancelled(id)) return Result.CANCELLED
                        received = true
                        output.append(pcm, "Direct TTS", model)
                    }
                    val reason = candidate.optString("finishReason")
                    if (reason.isNotEmpty()) {
                        check(reason == "STOP") { "Speech was incomplete" }
                        finished = true
                    }
                }
                if (cancelled(id)) return Result.CANCELLED
                check((finished || unary) && total > 0) { "Speech stream was truncated" }
                return Result.COMPLETE
            }
        } catch (failure: Exception) {
            return if (cancelled(id)) Result.CANCELLED else if (received) Result.PARTIAL
                else if (failure is IOException) Result.CONNECTION_ERROR
                else Result.UNAVAILABLE
        } finally {
            if (call === current) call = null
        }
    }

    private fun rememberFailure(model: String, credential: String, code: Int, body: String,
                                retryAfterSeconds: Long?) {
        val now = System.currentTimeMillis()
        val delay = PhoneVoiceCooldown.delayMillis(code, body, retryAfterSeconds, now)
        unavailableUntil[availabilityKey(model, credential)] = now + delay
    }

    private fun createTrack(): AudioTrack {
        val minimum = AudioTrack.getMinBufferSize(RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        check(minimum > 0)
        val track = AudioTrack.Builder().setAudioAttributes(AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA).setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder().setSampleRate(RATE).setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build())
            .setBufferSizeInBytes(maxOf(minimum * 2, 9600)).setTransferMode(AudioTrack.MODE_STREAM).build()
        if (track.state != AudioTrack.STATE_INITIALIZED) { track.release(); error("Phone audio did not initialise") }
        runCatching { track.playbackParams = PlaybackParams().setSpeed(1.10f).setPitch(1f) }
        return track
    }

    fun stop(expectedPlayback: Long? = null) {
        synchronized(turnGate) {
            if (expectedPlayback != null && playbackEpoch != expectedPlayback) return
            incrementalTurn = null
            speechQueued = false
            val stoppedAt = generation.incrementAndGet()
            val previous = call
            val output = player
            if (player === output) player = null
            live.reset()
            deviceSpeech.stop()
            Thread({ previous?.cancel() }, "orion-phone-voice-stop").apply { isDaemon = true }.start()
            val finish = {
                runCatching { output?.pause(); output?.flush(); output?.stop() }
                if (generation.get() == stoppedAt) state(false)
            }
            if (Looper.myLooper() == Looper.getMainLooper()) finish() else main.post(finish)
        }
    }

    fun close() {
        closed = true
        synchronized(instances) { instances.remove(this) }
        stop()
        worker.shutdownNow()
        live.close()
        deviceSpeech.close()
        closeHttpResources(client)
    }
}
