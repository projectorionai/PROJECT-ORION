package com.orion.companion

import java.io.DataInputStream
import java.io.InputStream

/** Bounded PCM frames; EOF without the zero-length end marker is a failure. */
class PcmFrames(input: InputStream, private val limit: Int = 24_000 * 2 * 180) {
    private val stream = DataInputStream(input)
    var totalBytes = 0
        private set
    private var finished = false

    fun next(): ByteArray? {
        check(!finished) { "Speech stream has already ended" }
        val size = stream.readInt()
        if (size == 0) {
            check(totalBytes > 0) { "Speech stream contained no audio" }
            finished = true
            return null
        }
        require(size in 2..65_536 && size % 2 == 0 && size <= limit - totalBytes) {
            "Invalid speech frame"
        }
        val bytes = ByteArray(size)
        stream.readFully(bytes)
        totalBytes += size
        return bytes
    }
}
