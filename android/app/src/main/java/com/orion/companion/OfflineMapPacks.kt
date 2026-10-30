package com.orion.companion

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import android.provider.OpenableColumns
import org.mapsforge.map.reader.MapFile
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Owns only the two offline vector map files; downloads never alter a usable pack in place. */
internal class OfflineMapPacks(context: Context) {
    enum class Pack(val label: String, val fileName: String, val url: String, val approximateSize: String) {
        GREAT_BRITAIN("Great Britain", "great-britain.map",
            "https://download.mapsforge.org/maps/v5/europe/great-britain.map", "about 1.5 GB"),
        IRELAND_NI("Ireland and Northern Ireland", "ireland-and-northern-ireland.map",
            "https://download.mapsforge.org/maps/v5/europe/ireland-and-northern-ireland.map", "about 323 MB")
    }

    data class Progress(val pack: Pack, val bytes: Long, val totalBytes: Long?, val phase: String)

    private val app = context.applicationContext
    // Also allows a prepared .map file to be placed with ADB under
    // /sdcard/Android/data/com.orion.companion/files/maps/.
    private val directory = app.getExternalFilesDir("maps") ?: File(app.filesDir, "offline-maps")
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "orion-offline-map-pack").apply { isDaemon = true }
    }
    private val main = Handler(Looper.getMainLooper())
    private val cancelled = AtomicBoolean(false)
    private val lock = Any()
    private var busy = false

    fun file(pack: Pack): File = File(directory, pack.fileName)

    fun installed(pack: Pack): Boolean {
        val candidate = file(pack)
        if (!candidate.isFile || candidate.length() < MIN_PACK_BYTES) return false
        return runCatching { validate(candidate, pack) }.isSuccess
    }

    fun installedFiles(): List<File> = Pack.entries.filter(::installed).map(::file)

    fun cancel() { cancelled.set(true) }

    fun close() { cancel(); worker.shutdown() }

    fun download(pack: Pack, progress: (Progress) -> Unit, done: (Result<File>) -> Unit) {
        start(pack, progress, done) { part, emit ->
            requireUnmeteredWiFi()
            val source = URL(pack.url)
            require(source.protocol == "https" && source.host == "download.mapsforge.org")
            val connection = (source.openConnection() as HttpURLConnection).apply {
                instanceFollowRedirects = false
                connectTimeout = 15_000
                readTimeout = 30_000
                setRequestProperty("User-Agent", "ORION-Android-offline-map/1")
            }
            try {
                connection.connect()
                if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                    throw IOException("Map server returned HTTP ${connection.responseCode}")
                }
                val expected = connection.contentLengthLong.takeIf { it > 0 }
                if (expected != null && expected > MAX_PACK_BYTES) throw IOException("Map exceeds the supported size")
                ensureSpace(expected ?: packMinimumSpace(pack))
                connection.inputStream.use { input -> copy(input, part, expected, emit, ::requireUnmeteredWiFi) }
            } finally {
                connection.disconnect()
            }
        }
    }

    fun import(pack: Pack, uri: Uri, progress: (Progress) -> Unit, done: (Result<File>) -> Unit) {
        start(pack, progress, done) { part, emit ->
            val expected = app.contentResolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use {
                if (it.moveToFirst() && !it.isNull(0)) it.getLong(0).takeIf { size -> size > 0 } else null
            }
            if (expected != null && expected > MAX_PACK_BYTES) throw IOException("Map exceeds the supported size")
            ensureSpace(expected ?: packMinimumSpace(pack))
            val input = app.contentResolver.openInputStream(uri) ?: throw IOException("The selected file cannot be opened")
            input.use { copy(it, part, expected, emit) }
        }
    }

    fun remove(pack: Pack, done: (Result<Unit>) -> Unit) {
        synchronized(lock) {
            if (busy) { main.post { done(Result.failure(IOException("Wait for the current transfer"))) }; return }
            busy = true
        }
        worker.execute {
            val result = runCatching {
                if (file(pack).exists() && !file(pack).delete()) throw IOException("Could not remove the map")
            }
            synchronized(lock) { busy = false }
            main.post { done(result) }
        }
    }

    private fun start(pack: Pack, progress: (Progress) -> Unit, done: (Result<File>) -> Unit,
        transfer: (File, (Long, Long?, String) -> Unit) -> Unit) {
        synchronized(lock) {
            if (busy) { main.post { done(Result.failure(IOException("Another map transfer is running"))) }; return }
            busy = true
            cancelled.set(false)
        }
        worker.execute {
            val result = runCatching {
                if (!directory.isDirectory && !directory.mkdirs()) throw IOException("Could not create map storage")
                val part = File(directory, "${pack.fileName}.part")
                if (part.exists() && !part.delete()) throw IOException("Could not clear an incomplete download")
                try {
                    var lastUpdate = 0L
                    val emit: (Long, Long?, String) -> Unit = { bytes, total, phase ->
                        val now = android.os.SystemClock.elapsedRealtime()
                        if (phase != "Copying" || now - lastUpdate >= 250 || (total != null && bytes == total)) {
                            lastUpdate = now
                            main.post { progress(Progress(pack, bytes, total, phase)) }
                        }
                    }
                    transfer(part, emit)
                    if (cancelled.get()) throw IOException("Transfer cancelled")
                    if (part.length() < MIN_PACK_BYTES) throw IOException("Map file is unexpectedly small")
                    emit(part.length(), part.length(), "Checking map")
                    validate(part, pack)
                    if (cancelled.get()) throw IOException("Transfer cancelled")
                    Files.move(part.toPath(), file(pack).toPath(), StandardCopyOption.REPLACE_EXISTING)
                    file(pack)
                } finally {
                    part.delete()
                }
            }
            synchronized(lock) { busy = false }
            main.post { done(result) }
        }
    }

    private fun copy(input: java.io.InputStream, part: File, expected: Long?,
        emit: (Long, Long?, String) -> Unit, transferCheck: () -> Unit = {}) {
        var copied = 0L
        val buffer = ByteArray(256 * 1024)
        FileOutputStream(part).use { fileOutput ->
            fileOutput.buffered(buffer.size).use { output ->
                while (true) {
                    if (cancelled.get()) throw IOException("Transfer cancelled")
                    val count = input.read(buffer)
                    if (count < 0) break
                    copied += count
                    if (copied > MAX_PACK_BYTES) throw IOException("Map exceeds the supported size")
                    if (copied % NETWORK_CHECK_INTERVAL < count) transferCheck()
                    if (copied % SPACE_CHECK_INTERVAL < count) ensureSpace(SPACE_RESERVE_BYTES)
                    output.write(buffer, 0, count)
                    emit(copied, expected, "Copying")
                }
                output.flush()
                fileOutput.fd.sync()
            }
        }
        if (expected != null && copied != expected) throw IOException("The map transfer ended before completion")
    }

    private fun requireUnmeteredWiFi() {
        val network = app.getSystemService(ConnectivityManager::class.java)
        val capabilities = network.getNetworkCapabilities(network.activeNetwork)
        if (capabilities == null || !capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED) ||
            !(capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) ||
                capabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET))) {
            throw IOException("Connect to unmetered Wi-Fi before downloading a large map")
        }
    }

    private fun ensureSpace(requiredBytes: Long) {
        if (!directory.isDirectory && !directory.mkdirs()) throw IOException("Could not create map storage")
        val free = StatFs(directory.absolutePath).availableBytes
        if (free < requiredBytes + SPACE_RESERVE_BYTES) {
            throw IOException("Not enough free phone storage for this map")
        }
    }

    private fun validate(part: File, pack: Pack) {
        val map = try { MapFile(part) } catch (error: Exception) {
            throw IOException("This is not a readable Mapsforge .map file", error)
        }
        try {
            val area = map.boundingBox()
            val matches = when (pack) {
                Pack.GREAT_BRITAIN -> area.contains(50.3755, -4.1427) &&
                    area.contains(52.4862, -1.8904) && area.contains(57.4778, -4.2247)
                Pack.IRELAND_NI -> area.contains(53.3498, -6.2603) && area.contains(54.5973, -5.9301)
            }
            if (!matches) throw IOException("The file does not appear to cover ${pack.label}")
        } finally {
            map.close()
        }
    }

    private fun packMinimumSpace(pack: Pack): Long = when (pack) {
        Pack.GREAT_BRITAIN -> 1_700_000_000L
        Pack.IRELAND_NI -> 450_000_000L
    }

    companion object {
        private const val MIN_PACK_BYTES = 64 * 1024L
        private const val MAX_PACK_BYTES = 3_000_000_000L
        private const val SPACE_RESERVE_BYTES = 256_000_000L
        private const val SPACE_CHECK_INTERVAL = 64 * 1024 * 1024L
        private const val NETWORK_CHECK_INTERVAL = 4 * 1024 * 1024L
    }
}
