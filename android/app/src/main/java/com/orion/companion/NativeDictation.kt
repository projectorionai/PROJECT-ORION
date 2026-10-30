package com.orion.companion

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.core.content.ContextCompat
import okhttp3.Call
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.sqrt

/** Foreground dictation uses the phone microphone directly, outside WebView audio. */
class NativeDictation(context: Context, private val onEvent: (JSONObject) -> Unit,
                      private val transcriber: ((ByteArray) -> String)? = null) {
    private class DictationFailure(val reason: String) : RuntimeException()
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val generation = AtomicLong()
    private val worker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-dictation").apply { isDaemon = true }
    }
    private val client = pinnedHttpClient(app).newBuilder()
        .connectTimeout(8, TimeUnit.SECONDS).readTimeout(85, TimeUnit.SECONDS)
        .callTimeout(90, TimeUnit.SECONDS).build()
    @Volatile private var closed = false
    @Volatile private var finishRequested = false
    @Volatile private var call: Call? = null
    @Volatile var busy = false
        private set

    private fun emit(id: Long, state: String, text: String = "", level: Double = 0.0) {
        main.post {
            if (!closed && generation.get() == id) onEvent(JSONObject()
                .put("state", state).put("text", text).put("level", level))
        }
    }

    fun start() {
        if (closed || busy) return
        val id = generation.incrementAndGet()
        if (ContextCompat.checkSelfPermission(app, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED) {
            emit(id, "error", "Allow ORION's microphone permission in Android settings.")
            return
        }
        busy = true
        finishRequested = false
        PhonePerformance.route(app, "Microphone")
        worker.execute {
            try {
                val captureStarted = PhonePerformance.now()
                val pcm = capture(id)
                PhonePerformance.record(app, "capture", captureStarted)
                if (closed || generation.get() != id) return@execute
                if (pcm.size < 3200) {
                    emit(id, "error", "No speech was captured. Check the phone microphone and try again.")
                    return@execute
                }
                emit(id, "transcribing")
                val transcriptionStarted = PhonePerformance.now()
                val text = try { transcriber?.invoke(wav(pcm)) ?: transcribe(wav(pcm), id) }
                    finally { PhonePerformance.record(app, "transcription", transcriptionStarted) }
                if (text.isBlank()) emit(id, "error", "I did not hear any words. Please try again.")
                else emit(id, "transcript", text)
            } catch (failure: DictationFailure) {
                emit(id, "error", failure.reason)
            } catch (_: Exception) {
                emit(id, "error", "Dictation failed. Check microphone permission and the AI connection.")
            } finally {
                busy = false
                emit(id, "finished")
            }
        }
    }

    fun finish() { finishRequested = true }

    private fun capture(id: Long): ByteArray {
        if (ContextCompat.checkSelfPermission(app, Manifest.permission.RECORD_AUDIO) !=
            PackageManager.PERMISSION_GRANTED) {
            throw DictationFailure("Allow ORION's microphone permission in Android settings.")
        }
        val minimum = AudioRecord.getMinBufferSize(16000, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT)
        require(minimum > 0)
        val recorder = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, 16000,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, maxOf(minimum * 2, 4096))
        val audio = ByteArrayOutputStream(480_000)
        try {
            check(recorder.state == AudioRecord.STATE_INITIALIZED)
            recorder.startRecording()
            check(recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING)
            emit(id, "listening")
            val frame = ByteArray(640)
            val started = SystemClock.elapsedRealtime()
            var lastVoice = 0L
            var lastMeter = 0L
            var outputWasPlaying = false
            var outputBlockedUntil = 0L
            while (!closed && generation.get() == id && !finishRequested &&
                SystemClock.elapsedRealtime() - started < SpeechCaptureTiming.MAX_UTTERANCE_MILLIS) {
                val count = recorder.read(frame, 0, frame.size)
                if (count < 0) error("Microphone read failed")
                if (count == 0) continue
                val now = SystemClock.elapsedRealtime()
                val outputPlaying = PhoneVoice.anySpeaking() || RemoteSpeech.anySpeaking()
                if (outputWasPlaying && !outputPlaying) outputBlockedUntil = now + 3_000L
                outputWasPlaying = outputPlaying
                if (outputPlaying || now < outputBlockedUntil || NavigationAudioState.captureBlocked(now)) {
                    // A dictation clip must contain only the user's speech, including its pre-roll.
                    audio.reset(); lastVoice = 0L
                    continue
                }
                if (audio.size() + count > SpeechCaptureTiming.MAX_PCM_BYTES)
                    throw DictationFailure("The voice request exceeds thirty seconds. Please repeat a shorter request.")
                audio.write(frame, 0, count)
                var energy = 0.0
                for (index in 0 until count - 1 step 2) {
                    val value = ((frame[index].toInt() and 255) or
                        (frame[index + 1].toInt() shl 8)).toShort().toDouble() / 32768.0
                    energy += value * value
                }
                val level = sqrt(energy / maxOf(1, count / 2))
                if (level > 0.006) lastVoice = now
                if (now - lastMeter >= 80) { emit(id, "level", level = level); lastMeter = now }
                if (SpeechCaptureTiming.shouldFinish(started, lastVoice, now)) break
            }
            if (!finishRequested && SystemClock.elapsedRealtime() - started >= SpeechCaptureTiming.MAX_UTTERANCE_MILLIS)
                throw DictationFailure("The voice request exceeds thirty seconds. Please repeat a shorter request.")
            return if (lastVoice > 0) audio.toByteArray() else byteArrayOf()
        } finally {
            runCatching { recorder.stop() }
            recorder.release()
        }
    }

    private fun transcribe(audio: ByteArray, id: Long): String {
        val credentials = Prefs.credentials(app) ?: error("Pairing required")
        val endpoints = (listOfNotNull(Prefs.baseUrl(app)) + Prefs.candidateUrls(app)).distinct()
        for (endpoint in endpoints) {
            if (closed || generation.get() != id) return ""
            val base = endpoint.toHttpUrlOrNull() ?: continue
            if (base.scheme != "https") continue
            val body = JSONObject().put("device_id", credentials.first)
                .put("refresh_token", credentials.second).toString()
                .toRequestBody("application/json; charset=utf-8".toMediaType())
            val token = try {
                request(Request.Builder().url(base.newBuilder().encodedPath("/v1/auth/token")
                    .query(null).build()).post(body).build(), id).use {
                    if (!it.isSuccessful) null else
                        JSONObject(it.body?.string().orEmpty()).optString("access_token")
                }
            } catch (_: Exception) { null }
            if (token.isNullOrBlank()) continue
            val response = request(Request.Builder()
                .url(base.newBuilder().encodedPath("/api/transcribe").query(null).build())
                .header("Authorization", "Bearer $token")
                .post(audio.toRequestBody("audio/wav".toMediaType())).build(), id)
            response.use {
                val source = it.body?.source() ?: error("Empty transcription")
                if (!source.request(65537) && source.buffer.size <= 65536) {
                    val result = JSONObject(source.readUtf8())
                    if (!it.isSuccessful || !result.optBoolean("ok")) {
                        throw DictationFailure(result.optString("error", "PC transcription failed").take(180))
                    }
                    return result.optString("text").take(4000)
                }
                error("Oversized transcription")
            }
        }
        error("PC unreachable")
    }

    private fun request(request: Request, id: Long): okhttp3.Response {
        check(!closed && generation.get() == id)
        val current = client.newCall(request)
        call = current
        check(!closed && generation.get() == id)
        return current.execute()
    }

    private fun wav(pcm: ByteArray): ByteArray = ByteBuffer.allocate(44 + pcm.size)
        .order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray()); putInt(36 + pcm.size); put("WAVEfmt ".toByteArray())
            putInt(16); putShort(1); putShort(1); putInt(16000); putInt(32000)
            putShort(2); putShort(16); put("data".toByteArray()); putInt(pcm.size); put(pcm)
        }.array()

    fun close() {
        closed = true
        generation.incrementAndGet()
        finishRequested = true
        worker.shutdownNow()
        closeHttpResources(client) { call?.cancel() }
    }

    /** Cancel capture and upload before another microphone owner starts. */
    fun cancel() {
        generation.incrementAndGet()
        finishRequested = true
        call?.cancel()
    }
}
