package com.orion.companion

import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Request
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import okio.Buffer
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.net.SocketTimeoutException
import java.util.concurrent.TimeUnit

class PhoneMcpTransportTest {
    private fun message(request: Request): JSONObject {
        val buffer = Buffer()
        request.body!!.writeTo(buffer)
        return JSONObject(buffer.readUtf8())
    }

    private fun response(request: Request, code: Int, body: String, type: String = "application/json",
                         session: String? = null): Response = Response.Builder().request(request)
        .protocol(Protocol.HTTP_1_1).code(code).message("test")
        .apply { if (session != null) header("MCP-Session-Id", session) }
        .body(body.toResponseBody(type.toMediaType())).build()

    private fun result(request: Request, body: JSONObject, session: String? = null): Response = response(request,
        200, JSONObject().put("jsonrpc", "2.0").put("id", message(request).getLong("id"))
            .put("result", body).toString(), session = session)

    private fun modernTools() = JSONObject().put("resultType", "complete").put("tools", JSONArray())
    private fun deadline() = System.nanoTime() + TimeUnit.SECONDS.toNanos(6)
    private fun transport(handler: (Request) -> Response): PhoneMcpTransport = PhoneMcpTransport(
        "https://mcp.example.test/api/mcp".toHttpUrl(), "reviewed-token",
        OkHttpClient.Builder().retryOnConnectionFailure(false).followRedirects(false)
            .addInterceptor { handler(it.request()) }.build())

    private fun refusal(request: Request, code: Int, supported: List<String> = emptyList()) = response(request,
        400, JSONObject().put("jsonrpc", "2.0").put("id", message(request).getLong("id"))
            .put("error", JSONObject().put("code", code).put("message", "test")
                .put("data", JSONObject().put("supported", JSONArray(supported)))).toString())

    @Test fun modernDiscoveryUsesOneReadWithMatchingMetadataAndHeaders() {
        var requests = 0
        val client = transport { request ->
            requests++
            assertEquals("Bearer reviewed-token", request.header("Authorization"))
            assertEquals("application/json, text/event-stream", request.header("Accept"))
            assertEquals("tools/list", request.header("Mcp-Method"))
            assertEquals(PhoneMcpTransport.MODERN, request.header("MCP-Protocol-Version"))
            assertNull(request.header("MCP-Session-Id"))
            val metadata = message(request).getJSONObject("params").getJSONObject("_meta")
            assertEquals(PhoneMcpTransport.MODERN, metadata.getString("io.modelcontextprotocol/protocolVersion"))
            assertEquals(0, metadata.getJSONObject("io.modelcontextprotocol/clientCapabilities").length())
            result(request, modernTools())
        }
        assertEquals(0, client.discover(deadline()) {}.getJSONArray("tools").length())
        assertEquals(1, requests)
        assertNull(client.terminationRequest())
    }

    @Test fun legacyInitialisesThenPropagatesNegotiatedSessionAndVersion() {
        val methods = mutableListOf<String>()
        val client = transport { request ->
            val method = message(request).getString("method")
            methods.add(method)
            when (methods.size) {
                1 -> response(request, 400, "Session required", "text/plain")
                2 -> {
                    assertEquals("initialize", method)
                    assertNull(request.header("MCP-Protocol-Version"))
                    assertNull(request.header("MCP-Session-Id"))
                    result(request, JSONObject().put("protocolVersion", "2025-06-18")
                        .put("capabilities", JSONObject().put("tools", JSONObject())), "session-1")
                }
                3 -> {
                    assertFalse(message(request).has("id"))
                    assertEquals("session-1", request.header("MCP-Session-Id"))
                    assertEquals("2025-06-18", request.header("MCP-Protocol-Version"))
                    response(request, 202, "")
                }
                else -> {
                    assertEquals("session-1", request.header("MCP-Session-Id"))
                    assertEquals("2025-06-18", request.header("MCP-Protocol-Version"))
                    assertNull(request.header("Mcp-Method"))
                    assertFalse(message(request).getJSONObject("params").has("_meta"))
                    result(request, JSONObject().put("tools", JSONArray()))
                }
            }
        }
        client.discover(deadline()) {}
        assertEquals(listOf("tools/list", "initialize", "notifications/initialized", "tools/list"), methods)
        val deletion = client.terminationRequest()!!
        assertEquals("DELETE", deletion.method)
        assertEquals("session-1", deletion.header("MCP-Session-Id"))
        assertEquals("2025-06-18", deletion.header("MCP-Protocol-Version"))
        assertNull(client.terminationRequest())
    }

    @Test fun advertisedLegacyVersionCanBeSelectedWithoutIgnoringModernErrors() {
        var count = 0
        val client = transport { request ->
            count++
            when (count) {
                1 -> refusal(request, -32022, listOf("2025-03-26"))
                2 -> {
                    assertEquals("2025-03-26", message(request).getJSONObject("params").getString("protocolVersion"))
                    result(request, JSONObject().put("protocolVersion", "2025-03-26").put("capabilities", JSONObject()))
                }
                else -> response(request, 202, "")
            }
        }
        assertEquals(0, client.discover(deadline()) {}.getJSONArray("tools").length())
        assertEquals(3, count) // A resources-only server is not asked to list tools.
    }

    @Test fun modernCapabilityHeaderAndUnsupportedVersionErrorsDoNotTriggerLegacyProbe() {
        for (code in listOf(-32020, -32021, -32601, -32022)) {
            var count = 0
            val client = transport { request -> count++; refusal(request, code, listOf("1900-01-01")) }
            try { client.discover(deadline()) {}; fail("Unsupported server was accepted") } catch (_: IOException) { }
            assertEquals(1, count)
        }
    }

    @Test fun authFailuresAndMissingEndpointsNeverProbeAnotherProtocol() {
        for (status in listOf(401, 403, 404, 405, 429, 500)) {
            var count = 0
            val client = transport { request -> count++; response(request, status, "not available", "text/plain") }
            try { client.discover(deadline()) {}; fail("Failed endpoint was accepted") } catch (_: IOException) { }
            assertEquals(1, count)
        }
    }

    @Test fun numericStringAndFractionalIdsCannotMasqueradeAsTheCurrentRequest() {
        for (badId in listOf("1", 1.5, JSONObject.NULL)) {
            val client = transport { request -> response(request, 200,
                JSONObject().put("jsonrpc", "2.0").put("id", badId).put("result", modernTools()).toString()) }
            try { client.discover(deadline()) {}; fail("A mismatched ID was accepted") } catch (_: IOException) { }
        }
    }

    @Test fun ssePrimingAndProgressDoNotHideTheActualResult() {
        val client = transport { request ->
            val answer = JSONObject().put("jsonrpc", "2.0").put("id", message(request).getLong("id"))
                .put("result", modernTools())
            response(request, 200, ": keep alive\nid: start\ndata:\n\n" +
                "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/progress\",\"params\":{}}\n\n" +
                "data: $answer\n\n", "text/event-stream")
        }
        assertEquals(0, client.discover(deadline()) {}.getJSONArray("tools").length())
    }

    @Test fun changedToolListInvalidatesTheApprovedDefinitions() {
        var count = 0
        val client = transport { request ->
            count++
            val answer = JSONObject().put("jsonrpc", "2.0").put("id", message(request).getLong("id"))
                .put("result", modernTools())
            response(request, 200, "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n\n" +
                "data: $answer\n\n", "text/event-stream")
        }
        client.discover(deadline()) {}
        assertTrue(client.invalidated)
        try { client.call("write", JSONObject(), JSONObject()) {}; fail("Stale tools were called") } catch (_: IOException) { }
        assertEquals(1, count)
    }

    @Test fun incompleteModernResultsCannotBeReportedAsSuccessfulActions() {
        for (type in listOf("input_required", "pending", "")) {
            val client = transport { request -> result(request, JSONObject().put("resultType", type)) }
            try { client.discover(deadline()) {}; fail("Incomplete result was accepted") } catch (_: IOException) { }
        }
    }

    @Test fun toolCallsAreNeverReplayedAfterHttpFailure() {
        var count = 0
        val client = transport { request ->
            count++
            if (count == 1) result(request, modernTools()) else response(request, 503, "unavailable", "text/plain")
        }
        client.discover(deadline()) {}
        try { client.call("write", JSONObject(), JSONObject()) {}; fail("A failed action was accepted") }
        catch (_: IOException) { }
        assertEquals(2, count)
    }

    @Test fun invalidToolReceiptsCannotBecomeAFalseSuccessfulAction() {
        for (receipt in listOf(
            JSONObject().put("resultType", "complete"),
            JSONObject().put("resultType", "complete").put("content", JSONArray()).put("isError", "false"),
            JSONObject().put("resultType", "complete").put("content", JSONArray()).put("structuredContent", "text"),
        )) {
            var count = 0
            val client = transport { request -> count++; result(request, if (count == 1) modernTools() else receipt) }
            client.discover(deadline()) {}
            try { client.call("write", JSONObject(), JSONObject()) {}; fail("A malformed receipt was accepted") }
            catch (_: IOException) { }
            assertEquals(2, count)
        }
    }

    @Test fun customHeadersFollowNestedPropertyPathsAndEncodeUnicode() {
        val schema = JSONObject("""{"type":"object","properties":{"account":{"type":"object","properties":{
            "region":{"type":"string","x-mcp-header":"Region"},
            "count":{"type":"integer","x-mcp-header":"Count"},
            "enabled":{"type":"boolean","x-mcp-header":"Enabled"}}}}}""")
        val args = JSONObject("""{"account":{"region":" français ","count":42,"enabled":true}}""")
        val headers = PhoneMcpTransport.parameterHeaders(schema, args)
        assertEquals("42", headers["Mcp-Param-Count"])
        assertEquals("true", headers["Mcp-Param-Enabled"])
        assertTrue(headers["Mcp-Param-Region"]!!.startsWith("=?base64?"))
        assertEquals(emptyMap<String, String>(), PhoneMcpTransport.parameterHeaders(schema, JSONObject()))
    }

    @Test fun customHeaderSchemaRejectsDuplicateNamesUnsafeNamesArraysAndNumberTypes() {
        for (schema in listOf(
            """{"type":"object","properties":{"a":{"type":"string","x-mcp-header":"Same"},"b":{"type":"string","x-mcp-header":"same"}}}""",
            """{"type":"object","properties":{"a":{"type":"string","x-mcp-header":"bad header"}}}""",
            """{"type":"object","properties":{"a":{"type":"number","x-mcp-header":"Count"}}}""",
            """{"type":"object","properties":{"a":{"type":"array","items":{"type":"string","x-mcp-header":"Item"}}}}""",
            """{"type":"object","allOf":[{"properties":{"a":{"type":"string","x-mcp-header":"Composed"}}}]}""",
        )) try { PhoneMcpTransport.parameterHeaders(JSONObject(schema)); fail("An unsafe schema was accepted") }
        catch (_: IOException) { }
    }

    @Test fun primitiveTypesAndIntegerRangesAreCheckedBeforeTheCall() {
        val schema = JSONObject("""{"type":"object","properties":{"x":{"type":"integer","x-mcp-header":"Count"}}}""")
        for (value in listOf(1.5, "2", Long.MAX_VALUE)) try {
            PhoneMcpTransport.parameterHeaders(schema, JSONObject().put("x", value)); fail("An invalid integer was accepted")
        } catch (_: IOException) { }
        assertTrue(PhoneMcpTransport.parameterHeaders(schema, JSONObject().put("x", JSONObject.NULL)).isEmpty())
    }

    @Test fun modernToolRequestMirrorsItsHeadersWithoutChangingReviewedArguments() {
        var count = 0
        val arguments = JSONObject().put("region", "UK\nWest")
        val schema = JSONObject("""{"type":"object","properties":{"region":{"type":"string","x-mcp-header":"Region"}}}""")
        val client = transport { request ->
            count++
            if (count == 1) result(request, modernTools()) else {
                assertEquals("tools/call", request.header("Mcp-Method"))
                assertEquals("get_weather", request.header("Mcp-Name"))
                assertEquals(PhoneMcpTransport.headerValue("UK\nWest"), request.header("Mcp-Param-Region"))
                assertEquals(arguments.toString(), message(request).getJSONObject("params").getJSONObject("arguments").toString())
                result(request, JSONObject().put("resultType", "complete").put("content", JSONArray()))
            }
        }
        client.discover(deadline()) {}
        client.call("get_weather", arguments, schema) {}
        assertEquals(2, count)
    }

    @Test fun expiryOrCancellationStopsBeforeAnyNetworkRequest() {
        var count = 0
        val client = transport { request -> count++; result(request, modernTools()) }
        try { client.discover(System.nanoTime() - 1) {}; fail("Expired discovery ran") } catch (_: SocketTimeoutException) { }
        try { client.discover(deadline()) { throw IOException("cancelled") }; fail("Cancelled discovery ran") }
        catch (_: IOException) { }
        assertEquals(0, count)
    }

    @Test fun headerSentinelsCannotBeMistakenForPreEncodedValues() {
        assertEquals("simple", PhoneMcpTransport.headerValue("simple"))
        for (value in listOf(" padded ", "line\nvalue", "Hello 🌍", "=?base64?literal?="))
            assertNotEquals(value, PhoneMcpTransport.headerValue(value))
    }
}
