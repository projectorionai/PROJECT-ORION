package com.orion.companion

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.ConcurrentHashMap

/** Bounded phone-private GPX storage, shared by every map screen in this process. */
internal class OfflineRouteStore(file: File) {
    private val file = file.canonicalFile
    private val state = states.computeIfAbsent(this.file.path) { State() }

    private class State { var revision = 0L }

    /** Reserve at request time so an older calculation cannot replace a newer journey. */
    fun beginChange(): Long = synchronized(state) { ++state.revision }

    fun currentRevision(): Long = synchronized(state) { state.revision }

    fun isCurrent(revision: Long): Boolean = synchronized(state) { state.revision == revision }

    /** A navigation service may reserve a reroute only for the journey it is still following. */
    fun beginChangeIfCurrent(expectedRevision: Long): Long? = synchronized(state) {
        if (state.revision == expectedRevision) ++state.revision else null
    }

    /** Route and revision are read together so provenance cannot attach to a different journey. */
    fun readSnapshot(): Pair<OfflineRoute?, Long> = synchronized(state) { read() to state.revision }

    fun read(): OfflineRoute? = synchronized(state) {
        if (!file.exists()) return null
        if (!file.isFile || file.length() > OfflineRouteGpx.MAX_BYTES)
            throw IOException("The saved route is unreadable or exceeds 4 MB")
        val bytes = file.inputStream().use { input ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(8192)
            while (true) {
                if (Thread.currentThread().isInterrupted) throw IOException("Route restoration cancelled")
                val count = input.read(buffer)
                if (count < 0) break
                if (output.size() + count > OfflineRouteGpx.MAX_BYTES)
                    throw IOException("Saved GPX exceeds 4 MB")
                output.write(buffer, 0, count)
            }
            output.toByteArray()
        }
        OfflineRouteGpx.parse(OfflineRouteGpx.decodeUtf8(bytes))
    }

    fun save(xml: String, revision: Long, current: () -> Boolean = { true }): OfflineRoute {
        val parsed = OfflineRouteGpx.parse(xml)
        return synchronized(state) {
            fun checkCurrent() {
                if (state.revision != revision || !current() || Thread.currentThread().isInterrupted)
                    throw IOException("Route was superseded or cancelled")
            }
            checkCurrent()
            val directory = file.parentFile ?: throw IOException("Route storage has no parent directory")
            if (!directory.isDirectory && !directory.mkdirs()) throw IOException("Could not create route storage")
            val pending = File(directory, "${file.name}.pending")
            try {
                FileOutputStream(pending).use { output ->
                    output.write(xml.toByteArray(Charsets.UTF_8))
                    output.fd.sync()
                }
                checkCurrent()
                try {
                    Files.move(pending.toPath(), file.toPath(),
                        StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
                } catch (_: AtomicMoveNotSupportedException) {
                    Files.move(pending.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
                }
                parsed
            } finally { pending.delete() }
        }
    }

    /** A stale screen's delayed clear must not erase the latest requested route. */
    fun clear(revision: Long) = synchronized(state) {
        if (state.revision != revision) return@synchronized
        if (file.exists() && !file.delete()) throw IOException("Could not remove the saved route")
    }

    companion object {
        private val states = ConcurrentHashMap<String, State>()
    }
}
