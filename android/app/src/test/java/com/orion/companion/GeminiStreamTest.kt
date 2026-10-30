package com.orion.companion

import java.io.ByteArrayInputStream
import java.io.EOFException
import java.io.InputStream
import org.junit.Assert.*
import org.junit.Test

class GeminiStreamTest {
    @Test fun fragmentedUtf8AndMetadataReachTheConsumerUnchanged() {
        val json = "{\"text\":\"Hello 🌠\",\"thoughtSignature\":\"opaque==\",\"functionCall\":{\"name\":\"phone_task\"}}"
        val raw = "\uFEFF: keep-alive\r\ndata: $json\r\n\r\ndata: FINAL WORDS\n\n".toByteArray()
        val fragmented = object : InputStream() {
            var offset = 0
            override fun read() = if (offset == raw.size) -1 else raw[offset++].toInt() and 255
            override fun read(buffer: ByteArray, start: Int, length: Int): Int {
                val value = read()
                if (value < 0) return -1
                buffer[start] = value.toByte()
                return 1
            }
        }
        val events = mutableListOf<String>()
        GeminiStream.consume(fragmented) { events.add(it); true }
        assertEquals(listOf(json, "FINAL WORDS"), events)
    }

    @Test fun firstEventIsDeliveredBeforeReadingTheFollowingResponse() {
        val first = "data: first\n\n".toByteArray()
        var delivered = false
        val source = object : InputStream() {
            var offset = 0
            override fun read(): Int {
                if (offset == first.size) { assertTrue(delivered); return -1 }
                return first[offset++].toInt()
            }
            override fun read(buffer: ByteArray, start: Int, length: Int): Int {
                val value = read()
                if (value < 0) return -1
                buffer[start] = value.toByte()
                return 1
            }
        }
        GeminiStream.consume(source) { delivered = true; true }
        assertTrue(delivered)
    }

    @Test fun truncationAndOversizedResponsesDoNotBecomeCompletedReplies() {
        for (raw in listOf("data: unfinished", "data: unfinished\n")) {
            try { GeminiStream.consume(ByteArrayInputStream(raw.toByteArray())) { true }; fail("Truncation was accepted") }
            catch (_: EOFException) { }
        }
        try {
            GeminiStream.consume(ByteArrayInputStream("data: ${"x".repeat(64)}\n\n".toByteArray()), maxEventBytes = 16) { true }
            fail("An oversized event was accepted")
        } catch (_: IllegalArgumentException) { }
    }

    @Test fun multiLineDataAndEarlyCompletionAreSupported() {
        val events = mutableListOf<String>()
        GeminiStream.consume(ByteArrayInputStream("event: result\ndata: line one\ndata:line two\n\ndata: should not read\n\n".toByteArray())) {
            events.add(it); false
        }
        assertEquals(listOf("line one\nline two"), events)
    }

    @Test fun multiLineUtf8EventIsLimitedByBytes() {
        val raw = "data: é\ndata: é\ndata: é\ndata: é\n\n".toByteArray(Charsets.UTF_8)
        try {
            GeminiStream.consume(ByteArrayInputStream(raw), maxEventBytes = 8) { true }
            fail("A multi-line UTF-8 event exceeded its byte limit")
        } catch (_: IllegalArgumentException) { }
    }
}
