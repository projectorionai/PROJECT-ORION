package com.orion.companion

import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Bounded SSE decoding shared by direct phone speech providers. */
internal class PhoneSseEvents(
    stream: InputStream,
    private val maxEventChars: Int = 2_097_152,
    private val maxStreamChars: Int = 16_777_216,
) {
    private val input = BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 8192)
    private var total = 0
    private var firstLine = true
    private var events = 0

    fun next(): String? {
        val data = StringBuilder()
        while (true) {
            val line = line() ?: run {
                check(data.isEmpty()) { "Speech stream ended inside an event" }
                return null
            }
            if (line.isEmpty()) {
                if (data.isNotEmpty()) {
                    check(++events <= 4096) { "Too many speech events" }
                    return data.toString()
                }
            } else if (line.startsWith("data:")) {
                val value = line.substring(5).removePrefix(" ")
                if (data.isNotEmpty()) data.append('\n')
                check(data.length + value.length <= maxEventChars) { "Speech event is too large" }
                data.append(value)
            }
        }
    }

    private fun line(): String? {
        val out = StringBuilder()
        while (true) {
            val ch = input.read()
            if (ch < 0) {
                if (out.isEmpty()) return null
                // SSE events need a blank separator, including the final event.
                throw IllegalStateException("Speech stream ended inside a line")
            }
            check(++total <= maxStreamChars) { "Speech stream is too large" }
            if (ch == '\n'.code) break
            check(out.length < maxEventChars) { "Speech line is too large" }
            out.append(ch.toChar())
        }
        if (out.isNotEmpty() && out.last() == '\r') out.setLength(out.length - 1)
        var text = out.toString()
        if (firstLine) { text = text.removePrefix("\uFEFF"); firstLine = false }
        return text
    }
}

/** Parses an optional wake name and exact speech control commands. */
internal object PhoneWakeWords {
    // Speech models sometimes punctuate the name or hear its final vowel as
    // "Orian". Keep the trigger anchored at the start of the utterance.
    private val wake = Regex("^[\\s\\\"'‘’“”]*?(?:(?:hey|okay|ok|hello)[,\\s]+)?(?:orion|orian|o[.\\s-]*r[.\\s-]*i[.\\s-]*o[.\\s-]*n)(?:[\\s,:.!?—-]+(.*))?$",
        RegexOption.IGNORE_CASE)
    private val stop = Regex("^(?:stop|pause|quiet|stop speaking|be quiet)[.!?]*$", RegexOption.IGNORE_CASE)
    fun request(transcript: String): String? = wake.matchEntire(transcript.trim().trimEnd('"', '\'', '’', '”'))
        ?.groupValues?.get(1)?.trim()?.trimEnd('"', '\'', '’', '”')
    fun isStop(transcript: String): Boolean = request(transcript)?.let { stop.matches(it) } == true
    fun isConversationStop(transcript: String): Boolean =
        stop.matches(request(transcript) ?: transcript.trim())
    fun conversationRequest(transcript: String): String = request(transcript) ?: transcript.trim()
    fun stopListening(request: String): Boolean = Regex("^(?:stop listening|turn listening off)[.!?]*$",
        RegexOption.IGNORE_CASE).matches(request.trim())
    fun retryDelay(failures: Int): Long = (1000L shl failures.coerceIn(1, 6)).coerceAtMost(60_000L)
    fun wav(pcm: ByteArray): ByteArray {
        require(pcm.isNotEmpty() && pcm.size <= SpeechCaptureTiming.MAX_PCM_BYTES && pcm.size % 2 == 0)
        return ByteBuffer.allocate(44 + pcm.size).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII)); putInt(36 + pcm.size)
            put("WAVEfmt ".toByteArray(Charsets.US_ASCII)); putInt(16)
            putShort(1); putShort(1); putInt(16_000); putInt(32_000)
            putShort(2); putShort(16); put("data".toByteArray(Charsets.US_ASCII)); putInt(pcm.size)
            put(pcm)
        }.array()
    }
}

/** Retains one consented frame in memory and refuses stale or oversized evidence. */
internal class PhoneFrameCache(private val maxBytes: Int = 524_288, private val maxAgeMs: Long = 10_000L) {
    private var jpeg: ByteArray? = null
    private var capturedAt = 0L
    @Synchronized fun put(frame: ByteArray, now: Long) {
        if (frame.size !in 4..maxBytes || frame[0] != 0xff.toByte() || frame[1] != 0xd8.toByte()) return
        jpeg = frame.copyOf(); capturedAt = now
    }
    @Synchronized fun get(now: Long): ByteArray? {
        if (now < capturedAt || now - capturedAt >= maxAgeMs) { clear(); return null }
        return jpeg?.copyOf()
    }
    @Synchronized fun clear() { jpeg = null; capturedAt = 0L }
}
