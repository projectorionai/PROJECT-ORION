package com.orion.companion

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.io.IOException
import java.io.ByteArrayInputStream

class PhoneHostedChatTest {
    private fun source(): JSONObject = JSONObject().put("contents", JSONArray()
        .put(JSONObject().put("role", "user").put("parts", JSONArray()
            .put(JSONObject().put("text", "Open Calculator")))))

    @Test fun deepSeekPhoneTurnsExplicitlyDisableThinking() {
        val request = PhoneHostedChat.request(source(), "deepseek-flash", HostedTextProviders.DEEPSEEK)
        assertEquals("disabled", request.getJSONObject("thinking").getString("type"))
        assertFalse(PhoneHostedChat.request(source(), "mistral-small-latest",
            HostedTextProviders.MISTRAL).has("thinking"))
    }

    @Test fun toolReceiptsRetainExactProviderCallIdentifiers() {
        val call = JSONObject().put("id", "abc123XYZ").put("name", "phone_action")
            .put("args", JSONObject().put("kind", "open_app"))
        val input = source()
        input.getJSONArray("contents").put(JSONObject().put("role", "model").put("parts",
            JSONArray().put(JSONObject().put("functionCall", call))))
            .put(JSONObject().put("role", "user").put("parts", JSONArray().put(JSONObject()
                .put("functionResponse", JSONObject().put("id", "abc123XYZ")
                    .put("response", JSONObject().put("ok", true))))))
        val messages = PhoneHostedChat.request(input, "mistral-small-latest").getJSONArray("messages")
        assertEquals("abc123XYZ", messages.getJSONObject(1).getJSONArray("tool_calls")
            .getJSONObject(0).getString("id"))
        assertEquals("abc123XYZ", messages.getJSONObject(2).getString("tool_call_id"))
    }

    @Test fun streamedTextIgnoresProviderReasoningFields() {
        val spoken = StringBuilder()
        val answer = PhoneHostedAnswer { spoken.append(it) }
        answer.accept("""{"choices":[{"delta":{"reasoning_content":"private analysis","content":"Hello"},"finish_reason":null}]}""")
        answer.accept("""{"choices":[{"delta":{},"finish_reason":"stop"}]}""")
        assertEquals("Hello", spoken.toString())
        assertEquals("Hello", answer.result().getJSONArray("candidates").getJSONObject(0)
            .getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text"))
    }

    @Test(expected = IOException::class) fun tokenTruncatedToolCallCannotExecute() {
        val answer = PhoneHostedAnswer {}
        answer.accept("""{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"abc123XYZ","function":{"name":"phone_action","arguments":"{}"}}]},"finish_reason":"length"}]}""")
        answer.result()
    }

    @Test fun tokenTruncatedTextRemainsAvailableForContinuation() {
        val answer = PhoneHostedAnswer {}
        answer.accept("""{"choices":[{"delta":{"content":"First part."},"finish_reason":"length"}]}""")
        assertEquals("MAX_TOKENS", answer.result().getJSONArray("candidates").getJSONObject(0)
            .getString("finishReason"))
    }

    @Test fun tokenSizedLongStreamsFinishBeyondTheSpeechEventLimit() {
        val transcript = StringBuilder()
        val events = buildString {
            append("data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n")
            repeat(4096) { append("data: {\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":null}]}\n\n") }
            append("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n")
            append("data: [DONE]\n\n")
        }
        val result = PhoneHostedAnswer { transcript.append(it) }
            .read(ByteArrayInputStream(events.toByteArray()))
        assertEquals(4096, transcript.length)
        assertEquals("MAX_TOKENS", result.getJSONArray("candidates").getJSONObject(0).getString("finishReason"))
    }

    @Test fun doneEndsTheReadWithoutWaitingForTheServerToCloseItsConnection() {
        val events = ("data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}\n\n" +
            "data: [DONE]\n\n").toByteArray()
        val stillOpen = object : ByteArrayInputStream(events) {
            override fun read(bytes: ByteArray, offset: Int, length: Int): Int {
                if (available() == 0) throw IOException("The connection is still open")
                return super.read(bytes, offset, length)
            }
        }
        val result = PhoneHostedAnswer {}.read(stillOpen)
        assertEquals("Hello", result.getJSONArray("candidates").getJSONObject(0)
            .getJSONObject("content").getJSONArray("parts").getJSONObject(0).getString("text"))
    }
}
