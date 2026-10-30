package com.orion.companion

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.UUID

internal data class OfflinePlace(val id: String, val name: String, val coordinate: OfflineCoordinate) {
    init {
        require(runCatching { UUID.fromString(id).toString() == id }.getOrDefault(false)) { "Invalid saved place ID" }
        require(name.isNotBlank() && name.length <= 80 && name.none { it.code < 32 }) { "A place name needs 1–80 readable characters" }
    }
}

/** Phone-private named coordinates; no address-service lookup, account or PC is involved. */
internal class OfflinePlaces(private val file: File) {
    fun list(): List<OfflinePlace> = synchronized(STORE_LOCK) {
        if (!file.exists()) return emptyList()
        if (!file.isFile || file.length() > MAX_BYTES) throw IOException("Saved places file is unreadable")
        try {
            return DataInputStream(file.inputStream().buffered()).use { input ->
                require(input.readInt() == MAGIC) { "Saved places format is unsupported" }
                val count = input.readInt()
                require(count in 0..MAX_PLACES) { "Saved places count is invalid" }
                val places = List(count) {
                    OfflinePlace(input.readUTF(), input.readUTF(),
                        OfflineCoordinate(input.readDouble(), input.readDouble()))
                }
                require(places.map { it.id }.distinct().size == places.size) { "Saved place IDs are duplicated" }
                require(input.read() == -1) { "Saved places file has trailing data" }
                places
            }
        } catch (error: Exception) { throw IOException("Saved places could not be read", error) }
    }

    fun save(name: String, coordinate: OfflineCoordinate): OfflinePlace = synchronized(STORE_LOCK) {
        val place = OfflinePlace(UUID.randomUUID().toString(), name.trim(), coordinate)
        val places = list()
        require(places.size < MAX_PLACES) { "Remove a saved place before adding another (limit 128)" }
        write(places + place)
        return place
    }

    fun remove(id: String): Unit = synchronized(STORE_LOCK) {
        val places = list()
        require(places.any { it.id == id }) { "This saved place no longer exists" }
        write(places.filterNot { it.id == id })
    }

    private fun write(places: List<OfflinePlace>) {
        file.parentFile?.let { if (!it.isDirectory && !it.mkdirs()) throw IOException("Could not create saved place storage") }
        val pending = File(file.parentFile, "${file.name}.pending")
        try {
            FileOutputStream(pending).use { output ->
                val data = DataOutputStream(output)
                data.writeInt(MAGIC)
                data.writeInt(places.size)
                places.forEach { place ->
                    data.writeUTF(place.id); data.writeUTF(place.name)
                    data.writeDouble(place.coordinate.latitude); data.writeDouble(place.coordinate.longitude)
                }
                data.flush()
                output.fd.sync()
            }
            try {
                Files.move(pending.toPath(), file.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(pending.toPath(), file.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
        } finally { pending.delete() }
    }

    companion object {
        // Two map activities or store instances must not lose each other's saved places.
        private val STORE_LOCK = Any()
        private const val MAGIC = 0x4f525031 // ORP1
        private const val MAX_BYTES = 128 * 1024L
        const val MAX_PLACES = 128
    }
}
