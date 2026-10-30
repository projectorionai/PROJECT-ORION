package com.orion.companion

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import java.io.Closeable
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Explicit phone voice and authorised fallback. Never repeats partially delivered cloud text. */
internal class PhoneDeviceSpeech(context: Context) : Closeable {
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val gate = Any()
    private val sequence = AtomicLong()
    private var engine: TextToSpeech? = null
    private var ready: CountDownLatch? = null
    private val turns = DeviceSpeechTurns()
    private var usable = false
    private var closed = false

    /** Called only from a speech worker; waits are cancellable and bounded. */
    fun speak(text: String, cancelled: () -> Boolean, firstSound: () -> Unit = {}): Boolean {
        if (cancelled()) return false
        val initialised = synchronized(gate) {
            if (closed) return false
            ready ?: CountDownLatch(1).also { wait ->
                ready = wait
                main.post {
                    synchronized(gate) {
                        if (closed) { wait.countDown(); return@post }
                        engine = TextToSpeech(app) { status -> main.post {
                            synchronized(gate) {
                                val tts = engine
                                usable = !closed && status == TextToSpeech.SUCCESS && tts != null &&
                                    tts.setLanguage(Locale.UK) !in listOf(
                                        TextToSpeech.LANG_MISSING_DATA, TextToSpeech.LANG_NOT_SUPPORTED)
                                if (usable) {
                                    tts?.setSpeechRate(1.10f)
                                    tts?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                                        override fun onStart(id: String) {
                                            turns.current?.takeIf { it.token == id }?.firstSound?.invoke()
                                        }
                                        override fun onDone(id: String) = complete(id, true)
                                        @Deprecated("Required by older Android speech engines")
                                        override fun onError(id: String) = complete(id, false)
                                        override fun onError(id: String, code: Int) = complete(id, false)
                                        override fun onStop(id: String, interrupted: Boolean) = complete(id, false)
                                    })
                                }
                                wait.countDown()
                            }
                        } }
                    }
                }
            }
        }
        if (!await(initialised, 10_000L, cancelled)) return false
        val utterance = DeviceSpeechTurn("orion-${sequence.incrementAndGet()}", firstSound)
        synchronized(gate) {
            if (closed || !usable || cancelled()) return false
            turns.begin(utterance)
        }
        main.post {
            synchronized(gate) {
                if (closed || turns.current !== utterance || cancelled() ||
                    engine?.speak(text, TextToSpeech.QUEUE_ADD, null, utterance.token) != TextToSpeech.SUCCESS)
                    complete(utterance.token, false)
            }
        }
        val finished = await(utterance.done, 180_000L, cancelled)
        if (!finished) stop()
        turns.clear(utterance)
        return finished && utterance.success && !cancelled()
    }

    private fun await(latch: CountDownLatch, timeout: Long, cancelled: () -> Boolean): Boolean {
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeout)
        while (!cancelled() && System.nanoTime() < deadline) {
            if (latch.await(50, TimeUnit.MILLISECONDS)) return true
        }
        return false
    }

    private fun complete(id: String, success: Boolean) = turns.complete(id, success)

    fun stop() {
        turns.cancel()
        main.post { synchronized(gate) { if (turns.current == null) engine?.stop() } }
    }

    override fun close() {
        synchronized(gate) { closed = true; ready?.countDown(); turns.cancel() }
        main.post { synchronized(gate) { engine?.stop(); engine?.shutdown(); engine = null } }
    }
}
