package com.orion.companion

import android.content.Context
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.Closeable
import java.io.IOException
import java.util.UUID
import java.util.concurrent.TimeUnit

/** Paired HTTPS PC controls. A power mutation is sent once to one authenticated host. */
internal class RemotePcPowerClient(context: Context) : Closeable {
    private class PcRejected(message: String) : IOException(message)
    private val app = context.applicationContext
    private val http = pinnedHttpClient(app).newBuilder()
        .connectTimeout(3, TimeUnit.SECONDS).readTimeout(5, TimeUnit.SECONDS)
        .callTimeout(7, TimeUnit.SECONDS).retryOnConnectionFailure(false)
        .followRedirects(false).followSslRedirects(false).build()
    private data class Connection(val base: HttpUrl, val credentials: Pair<String, String>,
                                  val token: String, val expiresAt: Long)
    private var connection: Connection? = null
    @Volatile private var closed = false

    private fun exchange(base: HttpUrl, path: String, body: JSONObject? = null,
                         token: String? = null, budgetNanos: Long = TimeUnit.SECONDS.toNanos(7)): JSONObject {
        if (closed) throw IOException("PC controls are closed.")
        val builder = Request.Builder().url(base.newBuilder().encodedPath(path).query(null).build())
        if (token != null) builder.header("Authorization", "Bearer $token")
        if (body == null) builder.get() else builder.post(body.toString()
            .toRequestBody("application/json; charset=utf-8".toMediaType()))
        val call = http.newCall(builder.build())
        call.timeout().timeout(budgetNanos.coerceIn(1, TimeUnit.SECONDS.toNanos(7)), TimeUnit.NANOSECONDS)
        call.execute().use { response ->
            val source = response.body?.source() ?: throw IOException("The PC returned an empty response.")
            if (source.request(16_385) || source.buffer.size > 16_384)
                throw IOException("The PC returned an invalid power response.")
            val data = try { JSONObject(source.readUtf8()) }
                catch (_: Exception) { throw IOException("The PC returned an invalid power response.") }
            if (!response.isSuccessful || !data.optBoolean("ok")) {
                if (response.code == 401) connection = null
                val detail = data.optString("error").takeIf { it.length in 1..300 }
                throw PcRejected(detail ?: "The PC refused this request (HTTP ${response.code}).")
            }
            return data
        }
    }

    /** Read-only discovery can try saved routes. Shutdown/cancel never fail over or retry. */
    fun status(): JSONObject {
        val credentials = Prefs.credentials(app)
            ?: throw IOException("Pair this phone with your PC to use PC controls.")
        val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(12)
        val cached = connection
        if (cached != null && cached.credentials == credentials && System.nanoTime() < cached.expiresAt) {
            try { return exchange(cached.base, "/api/pc/power", token = cached.token) }
            catch (_: IOException) { connection = null }
        }
        val routes = Prefs.candidateUrls(app).take(4)
        if (routes.isEmpty()) throw IOException("Set the current HTTPS address for your PC.")
        var problem = "The PC is unavailable. It must be running ORION and reachable at its saved HTTPS address."
        for (route in routes) {
            val remaining = deadline - System.nanoTime()
            if (remaining <= 0 || closed) break
            val base = Prefs.canonicalHttpsOrigin(route)?.toHttpUrlOrNull() ?: continue
            try {
                val tokenData = exchange(base, "/v1/auth/token", JSONObject()
                    .put("device_id", credentials.first).put("refresh_token", credentials.second),
                    budgetNanos = remaining)
                val token = tokenData.optString("access_token")
                if (token.length !in 1..1024) throw IOException("The PC returned an invalid pairing token.")
                val ttl = tokenData.optLong("expires_in", 900).coerceIn(1, 900)
                val connected = Connection(base, credentials, token,
                    System.nanoTime() + TimeUnit.SECONDS.toNanos((ttl - minOf(ttl / 2, 60))))
                val available = exchange(base, "/api/pc/power", token = token,
                    budgetNanos = deadline - System.nanoTime())
                connection = connected
                return available
            } catch (error: IOException) {
                // Show a bounded server reason if the route was reachable; no
                // URL, token or raw provider exception is placed in the UI.
                if (error.message?.startsWith("Pair ") == true ||
                    error.message?.contains("ledger") == true) problem = error.message.orEmpty()
            }
        }
        throw IOException(problem)
    }

    private fun connected(): Connection {
        val value = connection ?: throw IOException("Refresh the PC connection before this action.")
        if (Prefs.credentials(app) != value.credentials || System.nanoTime() >= value.expiresAt) {
            connection = null
            throw IOException("Refresh the PC connection and confirm the action again.")
        }
        return value
    }

    fun prepare(): JSONObject {
        val value = connected()
        return exchange(value.base, "/api/pc/power/prepare", JSONObject(), value.token)
    }

    fun shutdown(confirmation: String): JSONObject {
        val value = connected()
        return mutation(value, "/api/pc/power/shutdown", JSONObject()
            .put("request_id", UUID.randomUUID().toString()).put("confirmation", confirmation))
    }

    fun cancel(shutdownId: String): JSONObject {
        val value = connected()
        return mutation(value, "/api/pc/power/cancel", JSONObject()
            .put("request_id", UUID.randomUUID().toString()).put("shutdown_id", shutdownId))
    }

    private fun mutation(value: Connection, path: String, body: JSONObject): JSONObject = try {
        exchange(value.base, path, body, value.token)
    } catch (error: PcRejected) {
        throw error
    } catch (_: IOException) {
        throw IOException("The power request result is unknown. Refresh the PC connection before trying again.")
    }

    /** Explicit native pairing keeps the independent phone AI setting unchanged. */
    fun pair(rawOrigin: String, rawCode: String, rawCertificateCode: String = "") {
        val address = Prefs.parseAddress(rawOrigin)?.takeIf { it.https != false }
            ?: throw IOException("Enter your PC's full HTTPS address, without a path.")
        // A bare home Wi-Fi address means ORION's LAN link on port 8765.
        val port = address.port ?: if (LanTrust.isPrivateLanHost(address.host)) 8765 else Prefs.DEFAULT_PORT
        val hostPart = if (address.host.contains(':')) "[${address.host}]" else address.host
        val origin = Prefs.canonicalHttpsOrigin("https://$hostPart:$port/")?.toHttpUrlOrNull()
            ?: throw IOException("Enter your PC's full HTTPS address, without a path.")
        val code = rawCode.trim().uppercase(java.util.Locale.ROOT)
        if (!Regex("[A-Z0-9]{4}-[A-Z0-9]{4}").matches(code))
            throw IOException("Enter the fresh nine-character code from SECURITY → PAIR PHONE on the PC.")
        val lan = LanTrust.isPrivateLanHost(origin.host)
        if (lan) {
            // Home Wi-Fi: ORION's certificate is his own, so it must be the
            // one the PC shows. Read what this address presents (nothing is
            // sent) and compare it with the code the owner typed from the PC.
            val presented = try { LanTrust.presentedFingerprint(origin.host, origin.port) }
                catch (_: Exception) { throw IOException("The PC did not answer at ${origin.host}:${origin.port}. Check it is on this Wi-Fi.") }
            if (!LanTrust.codeMatches(presented, rawCertificateCode))
                throw IOException("The certificate code does not match this address. Nothing was sent. " +
                    "Type the 16-character certificate code shown beside the pairing code on the PC.")
            LanTrust.clearPins(app)
            if (!LanTrust.savePin(app, origin.host, origin.port, presented))
                throw IOException("The phone could not save the PC's certificate.")
        }
        val data = exchange(origin, "/v1/auth/pair", JSONObject()
            .put("code", code).put("device_name", "ORION Android"))
        if (!lan) LanTrust.clearPins(app)   // a new PC: its own LAN pin arrives with the endpoints
        val device = data.optString("device_id")
        val refresh = data.optString("refresh_token")
        if (device.length !in 1..64 || refresh.length !in 1..256)
            throw IOException("The PC returned an invalid pairing response.")
        Prefs.clearEndpoints(app)
        Prefs.clearCertPin(app)
        app.getSharedPreferences(Prefs.NAME, Context.MODE_PRIVATE).edit()
            .putString(Prefs.HOST, origin.host).putInt(Prefs.PORT, origin.port)
            .putBoolean(Prefs.HTTPS, true).commit()
        Prefs.saveCredentials(app, device, refresh)
        // Refuse a purported success if Android Keystore could not persist it.
        if (Prefs.credentials(app) != (device to refresh))
            throw IOException("The phone could not save its PC pairing securely.")
        data.optJSONArray("endpoints")?.let { Prefs.saveEndpoints(app, it.toString()) }
        connection = null
    }

    override fun close() {
        closed = true
        connection = null
        closeHttpResources(http)
    }
}
