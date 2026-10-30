package com.orion.companion

import android.content.Context
import okhttp3.Call
import okhttp3.HttpUrl
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.OkHttpClient
import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable
import java.io.IOException
import java.net.ConnectException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.UUID
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import javax.net.ssl.SSLException

/** A user-configured HTTPS MCP connection, independent of the PC and WebView. */
class PhoneMcpClient(context: Context) : Closeable {
    companion object {
        private const val FILE = "orion_phone_mcp"
        private val TOOL_NAME = Regex("[A-Za-z0-9_.-]{1,128}")
        private val DISCOVERY_BUDGET = TimeUnit.SECONDS.toNanos(6)
        private val FAILURE_COOLDOWN = TimeUnit.SECONDS.toNanos(30)
        private data class DiscoveryFailure(val revision: String, val atNanos: Long)
        @Volatile private var lastDiscoveryFailure: DiscoveryFailure? = null
        private fun prefs(context: Context) = context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
        private fun endpoint(value: String): HttpUrl? = value.toHttpUrlOrNull()?.takeIf {
            value.length <= 2048 && it.scheme == "https" && it.username.isEmpty() &&
                it.password.isEmpty() && it.query == null && it.fragment == null
        }
        fun save(context: Context, url: String, token: String): Boolean {
            val parsed = endpoint(url.trim()) ?: return false
            if (token.length > 4096 || token.any { it.code < 32 || it.code > 126 }) return false
            val sealed = if (token.isBlank()) null else PhoneSecretStore.encrypt(token) ?: return false
            lastDiscoveryFailure = null
            return prefs(context).edit().putString("url", parsed.toString()).putString("sealed_token", sealed)
                .putString("revision", UUID.randomUUID().toString()).putString("status", "Configured; discovery has not run.").commit()
        }
        fun clear(context: Context) { lastDiscoveryFailure = null; prefs(context).edit().clear().apply() }
        fun configured(context: Context): Boolean = endpoint(prefs(context).getString("url", "").orEmpty()) != null
        fun status(context: Context): String = if (!configured(context)) "No phone MCP server is configured."
            else prefs(context).getString("status", "Configured; discovery has not run.").orEmpty()
    }

    private data class Configuration(val url: HttpUrl, val token: String, val revision: String)
    private val app = context.applicationContext
    private val http = OkHttpClient.Builder().connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS).callTimeout(40, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false).followRedirects(false).followSslRedirects(false).build()
    private val generation = AtomicLong()
    @Volatile private var active: Call? = null
    @Volatile private var closed = false
    private var configuration: Configuration? = null
    @Volatile private var transport: PhoneMcpTransport? = null
    private var tools = JSONArray()
    private var discoveredAt = 0L

    fun configurationId(): String = prefs(app).getString("revision", "").orEmpty()
    fun serverLabel(): String = endpoint(prefs(app).getString("url", "").orEmpty())?.host ?: "unconfigured server"
    private fun load(): Configuration {
        val source = prefs(app)
        val url = endpoint(source.getString("url", "").orEmpty()) ?: throw IOException("Configure a phone MCP HTTPS endpoint first.")
        val sealed = source.getString("sealed_token", null)
        val token = if (sealed == null) "" else PhoneSecretStore.decrypt(sealed)
            ?: throw IOException("The phone MCP token cannot be opened. Enter it again.")
        return Configuration(url, token, source.getString("revision", "").orEmpty())
    }
    private fun check(id: Long) {
        if (closed || generation.get() != id) throw IOException("The phone MCP request was cancelled.")
    }

    @Synchronized fun discover(): JSONArray? {
        val id = generation.get()
        return try {
            check(id)
            val config = load()
            if (configuration?.revision == config.revision && transport?.invalidated == false && discoveredAt != 0L &&
                System.nanoTime() - discoveredAt < TimeUnit.MINUTES.toNanos(5)) return JSONArray(tools.toString())
            lastDiscoveryFailure?.let { failure ->
                if (failure.revision == config.revision &&
                    System.nanoTime() - failure.atNanos < FAILURE_COOLDOWN) return null
            }
            val deadline = System.nanoTime() + DISCOVERY_BUDGET
            releaseSession()
            configuration = config
            val connection = PhoneMcpTransport(config.url, config.token, http) { active = it }
            transport = connection
            val verify = {
                check(id)
                if (config.revision != configurationId()) throw IOException("The MCP configuration changed during this request.")
            }
            var result = connection.discover(deadline, verify)
            val found = JSONArray()
            val cursors = HashSet<String>()
            val names = HashSet<String>()
            var cursor = ""
            var excluded = 0
            for (page in 0 until 10) {
                val entries = result.optJSONArray("tools") ?: throw IOException("MCP discovery did not return a tool list.")
                if (entries.length() + found.length() + excluded > 100)
                    throw IOException("The MCP server exposes more than 100 tools; configure a smaller tool scope.")
                for (index in 0 until entries.length()) {
                    val entry = entries.optJSONObject(index) ?: throw IOException("MCP returned an invalid tool declaration.")
                    val name = entry.optString("name")
                    val schema = entry.optJSONObject("inputSchema") ?: throw IOException("MCP returned a tool without an input schema.")
                    if (!TOOL_NAME.matches(name) || entry.toString().length > 16_384 ||
                        !names.add(name))
                        throw IOException("MCP returned an invalid or repeated tool declaration.")
                    if (schema.opt("type") != "object")
                        throw IOException("MCP returned a tool with an invalid input schema.")
                    if (connection.protocol == PhoneMcpTransport.MODERN &&
                        runCatching { PhoneMcpTransport.parameterHeaders(schema) }.isFailure) {
                        excluded++
                        continue
                    }
                    found.put(JSONObject().put("name", name).put("description", entry.optString("description").take(1500))
                        .put("inputSchema", JSONObject(schema.toString())))
                }
                cursor = result.optString("nextCursor")
                if (cursor.isEmpty()) break
                if (cursor.length > 2048 || !cursors.add(cursor) || page == 9)
                    throw IOException("MCP discovery could not complete its pagination.")
                result = connection.list(cursor, deadline, verify)
            }
            check(id)
            if (config.revision != configurationId())
                throw IOException("The MCP configuration changed during discovery.")
            tools = found
            discoveredAt = System.nanoTime()
            lastDiscoveryFailure = null
            prefs(app).edit().putString("status", "Connected: ${found.length()} discovered tools " +
                "(${connection.protocol})." + if (excluded > 0) " $excluded tools excluded because their parameter headers are invalid." else "").apply()
            JSONArray(found.toString())
        } catch (failure: Exception) {
            tools = JSONArray(); discoveredAt = 0L
            releaseSession()
            val revision = configuration?.revision ?: configurationId()
            if (closed || generation.get() != id || revision != configurationId()) return null
            lastDiscoveryFailure = DiscoveryFailure(revision, System.nanoTime())
            val reason = when (failure) {
                is UnknownHostException -> "The MCP server could not be found."
                is SocketTimeoutException -> "The MCP server did not respond in time."
                is ConnectException -> "The MCP server refused the connection."
                is SSLException -> "The MCP server's secure connection failed."
                else -> when {
                    failure.message?.contains("HTTP 401") == true || failure.message?.contains("HTTP 403") == true ->
                        "The MCP server rejected its access token."
                    failure.message?.contains("unsupported protocol version") == true ->
                        "The MCP server uses an unsupported protocol version."
                    else -> "MCP discovery failed. Check the server and its configuration."
                }
            }
            prefs(app).edit().putString("status", "$reason Retrying after a short pause.").apply()
            null
        }
    }

    /** The caller obtains a fresh local approval before invoking this once. */
    @Synchronized fun call(tool: String, arguments: JSONObject, expectedConfiguration: String? = null): JSONObject {
        val id = generation.get()
        check(id)
        if (expectedConfiguration != null && expectedConfiguration != configurationId())
            throw IOException("The MCP endpoint changed after approval. Review the new endpoint first.")
        if (discoveredAt == 0L || configuration?.revision != configurationId())
            throw IOException("Discover this MCP server before calling one of its tools.")
        val definition = (0 until tools.length()).map { tools.getJSONObject(it) }.firstOrNull { it.optString("name") == tool }
            ?: throw IOException("That tool was not discovered on the configured MCP server.")
        if (arguments.toString().length > 32_768) throw IOException("The MCP arguments exceed the review limit.")
        val required = definition.getJSONObject("inputSchema").optJSONArray("required") ?: JSONArray()
        for (index in 0 until required.length()) if (!arguments.has(required.getString(index)))
            throw IOException("The MCP tool is missing a required argument.")
        val config = configuration ?: throw IOException("MCP is disconnected.")
        val result = (transport ?: throw IOException("MCP is disconnected.")).call(tool, arguments,
            definition.getJSONObject("inputSchema")) {
                check(id)
                if (config.revision != configurationId())
                    throw IOException("The MCP endpoint changed during this call. Its outcome is unverified.")
            }
        check(id)
        if (expectedConfiguration != null && expectedConfiguration != configurationId())
            throw IOException("The MCP endpoint changed during this call. Its outcome is unverified.")
        return boundedResult(result)
    }

    private fun releaseSession() {
        val old = transport ?: return
        transport = null
        // A dedicated dispatcher means session cleanup cannot cancel a new
        // request or interfere with a phone turn sharing the main HTTP client.
        Thread({
            val request = synchronized(this@PhoneMcpClient) { old.terminationRequest() } ?: return@Thread
            val cleanup = http.newBuilder().dispatcher(okhttp3.Dispatcher())
                .connectionPool(okhttp3.ConnectionPool(0, 1, TimeUnit.SECONDS))
                .connectTimeout(2, TimeUnit.SECONDS).readTimeout(2, TimeUnit.SECONDS)
                .callTimeout(2, TimeUnit.SECONDS).build()
            try { cleanup.newCall(request).execute().use { } }
            catch (_: Exception) { /* Best effort; the server may refuse DELETE. */ }
            finally { closeHttpResources(cleanup) }
        }, "orion-phone-mcp-session-close").apply { isDaemon = true }.start()
    }

    private fun boundedResult(result: JSONObject): JSONObject {
        val content = result.optJSONArray("content") ?: JSONArray()
        val safe = JSONArray()
        var remaining = 64_000
        var truncated = false
        for (index in 0 until minOf(content.length(), 50)) {
            val item = content.optJSONObject(index) ?: continue
            if (item.optString("type") == "text") {
                val text = item.optString("text")
                safe.put(JSONObject().put("type", "text").put("text", text.take(remaining)))
                if (text.length > remaining) truncated = true
                remaining = (remaining - text.length).coerceAtLeast(0)
            } else safe.put(JSONObject().put("type", item.optString("type")).put("detail", "Binary or linked content was returned; it was not automatically forwarded."))
        }
        val output = JSONObject().put("ok", !result.optBoolean("isError")).put("content", safe)
        result.optJSONObject("structuredContent")?.let {
            val text = it.toString()
            if (text.length <= remaining) output.put("structuredContent", it) else truncated = true
        }
        if (truncated || content.length() > 50) output.put("truncated", true)
        return output
    }

    fun cancel() {
        generation.incrementAndGet()
        val call = active ?: return
        // Cancelling a TLS call may close its socket. Keep that work off the
        // main thread when a phone turn is cancelled from the UI.
        Thread({ call.cancel() }, "orion-phone-mcp-cancel").apply { isDaemon = true }.start()
    }
    override fun close() { closed = true; cancel(); releaseSession(); closeHttpResources(http) }
}
