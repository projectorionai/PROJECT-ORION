package com.orion.companion

import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.EOFException
import java.io.InputStream

/** Bounded SSE decoding shared by direct Gemini and configured HTTP MCP. */
internal object GeminiStream {
    fun consume(
        input: InputStream,
        check: () -> Unit = {},
        maxEventBytes: Int = 1_048_576,
        maxTotalBytes: Int = 16_777_216,
        onEvent: (String) -> Boolean,
    ) {
        val source = BufferedInputStream(input)
        val line = ByteArrayOutputStream()
        val event = StringBuilder()
        var total = 0
        var firstLine = true
        var hasData = false
        var eventBytes = 0
        while (true) {
            if (total % 4096 == 0) check()
            val byte = source.read()
            if (byte < 0) {
                if (line.size() != 0 || hasData) throw EOFException("The stream ended inside an event.")
                return
            }
            require(++total <= maxTotalBytes) { "The stream exceeded its size limit." }
            if (byte != 10) {
                require(line.size() < maxEventBytes) { "An event line exceeded its size limit." }
                line.write(byte)
                continue
            }
            var text = line.toString(Charsets.UTF_8.name()).removeSuffix("\r")
            line.reset()
            if (firstLine) { text = text.removePrefix("\uFEFF"); firstLine = false }
            check()
            if (text.isEmpty()) {
                if (hasData) {
                    if (!onEvent(event.toString())) return
                    event.setLength(0)
                    hasData = false
                    eventBytes = 0
                }
            } else if (text == "data" || text.startsWith("data:")) {
                val value = if (text == "data") "" else text.substring(5).removePrefix(" ")
                val addition = value.toByteArray(Charsets.UTF_8).size + if (hasData) 1 else 0
                require(eventBytes + addition <= maxEventBytes) { "An event exceeded its size limit." }
                if (hasData) event.append('\n')
                event.append(value)
                hasData = true
                eventBytes += addition
            }
        }
    }
}
