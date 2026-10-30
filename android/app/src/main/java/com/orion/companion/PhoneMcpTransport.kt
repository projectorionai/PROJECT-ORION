package com.orion.companion

import okhttp3.Call
import okhttp3.HttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.Base64
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong

/** Bounded Streamable HTTP transport. Only read-only discovery can negotiate again. */
internal class PhoneMcpTransport(
    private val url: HttpUrl,
    private val token: String,
    private val http: OkHttpClient,
    private val activeCall: (Call?) -> Unit = {},
) {
    companion object {
        const val MODERN = "2026-07-28"
        val LEGACY = listOf("2025-11-25", "2025-06-18", "2025-03-26")
        private val HEADER_NAME = Regex("[!#$%&'*+.^_`|~0-9A-Za-z-]+")
        private const val RESPONSE_LIMIT = 1_048_576L
        private const val SAFE_INTEGER = 9_007_199_254_740_991L

        private fun clientInfo() = JSONObject().put("name", "ORION Android MCP transport").put("version", "1.0.0")

        /** Encode whitespace, Unicode, control characters and literal sentinel values. */
        fun headerValue(value: String): String = if (
            value.all { it.code in 32..126 || it == '\t' } && value == value.trim() &&
            !(value.startsWith("=?base64?") && value.endsWith("?="))
        ) value else "=?base64?${Base64.getEncoder().encodeToString(value.toByteArray(Charsets.UTF_8))}?="

        /** Reject unsafe schema annotations before advertising a tool to the model. */
        fun parameterHeaders(schema: JSONObject, arguments: JSONObject? = null): Map<String, String> {
            val names = HashSet<String>()
            val result = linkedMapOf<String, String>()
            var visited = 0
            fun visit(node: Any?, path: List<String>, reachable: Boolean, depth: Int) {
                if (++visited > 2000 || depth > 32) throw IOException("The MCP schema is too complex.")
                when (node) {
                    is JSONArray -> for (i in 0 until node.length()) visit(node.opt(i), path, false, depth + 1)
                    is JSONObject -> {
                        if (node.has("x-mcp-header")) {
                            val name = node.opt("x-mcp-header") as? String
                                ?: throw IOException("An MCP parameter header is invalid.")
                            val type = node.opt("type") as? String
                            if (!reachable || path.isEmpty() || !HEADER_NAME.matches(name) || name.length > 128 ||
                                type !in setOf("string", "integer", "boolean") || !names.add(name.lowercase()))
                                throw IOException("An MCP parameter header is invalid.")
                            var value: Any? = arguments
                            for (part in path) value = (value as? JSONObject)?.opt(part)
                            if (value != null && value != JSONObject.NULL) {
                                val rendered = when (type) {
                                    "string" -> (value as? String)
                                    "boolean" -> (value as? Boolean)?.toString()
                                    "integer" -> when (value) {
                                        is Byte, is Short, is Int, is Long -> (value as Number).toLong()
                                            .takeIf { it in -SAFE_INTEGER..SAFE_INTEGER }?.toString()
                                        else -> null
                                    }
                                    else -> null
                                } ?: throw IOException("An MCP parameter does not match its header type.")
                                if (rendered.toByteArray(Charsets.UTF_8).size > 4096)
                                    throw IOException("An MCP parameter header exceeds its size limit.")
                                result["Mcp-Param-$name"] = headerValue(rendered)
                            }
                        }
                        val keys = node.keys()
                        while (keys.hasNext()) {
                            val key = keys.next()
                            val child = node.opt(key)
                            if (key == "properties" && reachable && child is JSONObject) {
                                val properties = child.keys()
                                while (properties.hasNext()) {
                                    val property = properties.next()
                                    visit(child.opt(property), path + property, true, depth + 1)
                                }
                            } else if (child is JSONObject || child is JSONArray)
                                visit(child, path, false, depth + 1)
                        }
                    }
                }
            }
            visit(schema, emptyList(), true, 0)
            if (result.entries.sumOf { it.key.length + it.value.length } > 16_384)
                throw IOException("The MCP parameter headers exceed their size limit.")
            return result
        }
    }

    private class HttpFailure(val status: Int, val errorCode: Int?, val supported: List<String>) :
        IOException("MCP refused the request (HTTP $status). It has not been replayed.")

    private val ids = AtomicLong()
    private var session = ""
    var protocol: String = MODERN
        private set
    @Volatile var invalidated = false
        private set

    /** Probe with tools/list once; legacy fallback never occurs for an action. */
    fun discover(deadlineNanos: Long, check: () -> Unit): JSONObject {
        protocol = MODERN
        try {
            return rpc("tools/list", JSONObject(), deadlineNanos, check)
        } catch (failure: HttpFailure) {
            val legacyVersion = when {
                failure.status == 400 && failure.errorCode == -32022 ->
                    LEGACY.firstOrNull { it in failure.supported }
                        ?: throw IOException("The MCP server uses an unsupported protocol version.")
                failure.status == 400 && failure.errorCode !in setOf(-32020, -32021, -32601) -> LEGACY.first()
                else -> throw failure
            }
            protocol = legacyVersion
        }
        val initialise = rpc("initialize", JSONObject().put("protocolVersion", protocol)
            .put("capabilities", JSONObject()).put("clientInfo", clientInfo()), deadlineNanos, check)
        val negotiated = initialise.opt("protocolVersion") as? String
        if (negotiated !in LEGACY) throw IOException("The MCP server uses an unsupported protocol version.")
        protocol = negotiated!!
        val capabilities = initialise.optJSONObject("capabilities")
            ?: throw IOException("MCP did not negotiate server capabilities.")
        rpc("notifications/initialized", JSONObject(), deadlineNanos, check, notification = true)
        // A resources-only server is valid, but offers no tools to this client.
        if (capabilities.optJSONObject("tools") == null) return JSONObject().put("tools", JSONArray())
        return rpc("tools/list", JSONObject(), deadlineNanos, check)
    }

    fun list(cursor: String, deadlineNanos: Long, check: () -> Unit): JSONObject =
        rpc("tools/list", JSONObject().put("cursor", cursor), deadlineNanos, check)

    /** Never retries a tool call, including after an expired session or protocol error. */
    fun call(name: String, arguments: JSONObject, schema: JSONObject, check: () -> Unit): JSONObject {
        if (invalidated) throw IOException("The MCP tool list changed. Discover and review the tools again.")
        val headers = if (protocol == MODERN) parameterHeaders(schema, arguments) else emptyMap()
        val result = rpc("tools/call", JSONObject().put("name", name).put("arguments", arguments),
            System.nanoTime() + TimeUnit.SECONDS.toNanos(40), check, extraHeaders = headers)
        if (result.optJSONArray("content") == null ||
            (result.has("isError") && result.opt("isError") !is Boolean) ||
            (result.has("structuredContent") && result.optJSONObject("structuredContent") == null))
            throw IOException("MCP returned an invalid tool receipt. Its outcome is unverified; it has not been replayed.")
        return result
    }

    private fun rpc(method: String, params: JSONObject, deadline: Long, check: () -> Unit,
                    notification: Boolean = false, extraHeaders: Map<String, String> = emptyMap()): JSONObject {
        check()
        val remaining = deadline - System.nanoTime()
        if (remaining <= 0) throw SocketTimeoutException("The MCP request did not finish in time.")
        val requestId = ids.incrementAndGet()
        val payloadParams = JSONObject(params.toString())
        if (protocol == MODERN) payloadParams.put("_meta", JSONObject()
            .put("io.modelcontextprotocol/protocolVersion", protocol)
            .put("io.modelcontextprotocol/clientInfo", clientInfo())
            .put("io.modelcontextprotocol/clientCapabilities", JSONObject()))
        val message = JSONObject().put("jsonrpc", "2.0").put("method", method).put("params", payloadParams)
        if (!notification) message.put("id", requestId)
        val request = Request.Builder().url(url).header("Accept", "application/json, text/event-stream")
            .apply {
                if (method != "initialize") header("MCP-Protocol-Version", protocol)
                if (protocol == MODERN) {
                    header("Mcp-Method", method)
                    if (method == "tools/call") header("Mcp-Name", headerValue(params.getString("name")))
                    for ((name, value) in extraHeaders) header(name, value)
                }
                if (token.isNotEmpty()) header("Authorization", "Bearer $token")
                if (protocol != MODERN && session.isNotEmpty()) header("MCP-Session-Id", session)
            }.post(message.toString().toRequestBody("application/json; charset=utf-8".toMediaType())).build()
        val call = http.newCall(request)
        call.timeout().timeout(remaining, TimeUnit.NANOSECONDS)
        activeCall(call)
        try {
            check()
            call.execute().use { response ->
                if (!response.isSuccessful) {
                    if (response.code == 404 && session.isNotEmpty()) { session = ""; invalidated = true }
                    val raw = response.peekBody(16_384).string()
                    val envelope = runCatching { JSONObject(raw) }.getOrNull()
                    val error = envelope?.takeIf { it.opt("jsonrpc") == "2.0" &&
                        (!it.has("id") || it.isNull("id") || matchesId(it.opt("id"), requestId)) }?.optJSONObject("error")
                    val supported = error?.optJSONObject("data")?.optJSONArray("supported")
                    throw HttpFailure(response.code, (error?.opt("code") as? Number)?.toInt(),
                        if (supported == null) emptyList() else (0 until minOf(20, supported.length()))
                            .mapNotNull { supported.opt(it) as? String })
                }
                if (notification) {
                    if (response.code != 202) throw IOException("MCP did not acknowledge its initialised notification.")
                    check()
                    return JSONObject()
                }
                if (response.code == 202 || response.code == 204)
                    throw IOException("MCP accepted the request without a result. Its outcome is unverified; it has not been replayed.")
                val body = response.body ?: throw IOException("MCP returned no result.")
                var answer: JSONObject? = null
                fun accept(raw: String): Boolean {
                    val payload = try { JSONObject(raw) } catch (_: Exception) {
                        throw IOException("MCP returned an invalid JSON response.")
                    }
                    if (payload.opt("jsonrpc") != "2.0") throw IOException("MCP returned an invalid response.")
                    if (payload.has("method")) {
                        if (payload.has("id")) throw IOException("The MCP server requested an unsupported client capability.")
                        if (payload.opt("method") == "notifications/tools/list_changed") invalidated = true
                        return true
                    }
                    if (!matchesId(payload.opt("id"), requestId))
                        throw IOException("MCP returned a result for a different request.")
                    if (payload.has("error")) throw IOException("MCP reported a tool or protocol error. The request has not been replayed.")
                    val result = payload.optJSONObject("result") ?: throw IOException("MCP did not return a result object.")
                    val type = result.opt("resultType")
                    if (protocol == MODERN && type != "complete")
                        throw IOException("MCP returned an incomplete or unsupported result. Its outcome is unverified; it has not been replayed.")
                    answer = result
                    return false
                }
                when (body.contentType()?.let { "${it.type}/${it.subtype}" }) {
                    "text/event-stream" -> GeminiStream.consume(body.byteStream(), check = check,
                        maxTotalBytes = 2_097_152) {
                            // Legacy servers can prime a resumable SSE stream
                            // with an event ID and empty data before any reply.
                            if (it.isEmpty()) true else accept(it)
                        }
                    "application/json" -> {
                        val source = body.source()
                        if (source.request(RESPONSE_LIMIT + 1) || source.buffer.size > RESPONSE_LIMIT)
                            throw IOException("The MCP response exceeded its size limit.")
                        accept(source.readUtf8())
                    }
                    else -> throw IOException("MCP returned an unsupported response content type.")
                }
                val result = answer ?: throw IOException("No MCP result arrived. Its outcome is unverified; it has not been replayed.")
                if (method == "initialize") response.header("MCP-Session-Id")?.let {
                    if (it.length !in 1..1024 || it.any { character -> character.code !in 33..126 })
                        throw IOException("MCP returned an invalid session identifier.")
                    session = it
                }
                check()
                return result
            }
        } finally { activeCall(null) }
    }

    private fun matchesId(value: Any?, expected: Long): Boolean = when (value) {
        is Byte, is Short, is Int, is Long -> (value as Number).toLong() == expected
        else -> false
    }

    /** Build the optional legacy session DELETE; the owner runs it off the UI thread. */
    fun terminationRequest(): Request? {
        val current = session
        session = ""
        if (current.isEmpty() || protocol == MODERN) return null
        return Request.Builder().url(url).header("MCP-Session-Id", current)
            .header("MCP-Protocol-Version", protocol)
            .apply { if (token.isNotEmpty()) header("Authorization", "Bearer $token") }.delete().build()
    }
}
