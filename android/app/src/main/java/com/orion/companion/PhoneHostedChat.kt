package com.orion.companion

import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.io.InputStream

/** Converts ORION's reviewed Gemini-shaped action turn into Chat Completions. */
internal object PhoneHostedChat {
    fun request(source: JSONObject, model: String, provider: String = ""): JSONObject {
        val messages = JSONArray()
        val instruction = source.optJSONObject("systemInstruction")?.optJSONArray("parts")
        if (instruction != null) messages.put(JSONObject().put("role", "system")
            .put("content", text(instruction)))
        val contents = source.optJSONArray("contents") ?: JSONArray()
        for (turn in 0 until contents.length()) {
            val content = contents.optJSONObject(turn) ?: continue
            val parts = content.optJSONArray("parts") ?: JSONArray()
            val responses = (0 until parts.length()).mapNotNull { parts.optJSONObject(it)?.optJSONObject("functionResponse") }
            if (responses.isNotEmpty()) {
                for (response in responses) {
                    val id = response.optString("id")
                    if (id.isBlank()) throw IOException("The AI returned an action without its receipt ID.")
                    messages.put(JSONObject().put("role", "tool").put("tool_call_id", id)
                        .put("content", response.optJSONObject("response")?.toString() ?: "{}"))
                }
                continue
            }
            val calls = JSONArray()
            for (index in 0 until parts.length()) {
                val call = parts.optJSONObject(index)?.optJSONObject("functionCall") ?: continue
                val id = call.optString("id")
                if (id.isBlank()) throw IOException("The AI returned an action without its call ID.")
                calls.put(JSONObject().put("id", id).put("type", "function")
                    .put("function", JSONObject().put("name", call.optString("name"))
                        .put("arguments", call.optJSONObject("args")?.toString() ?: "{}")))
            }
            if (calls.length() > 0) {
                val assistant = JSONObject().put("role", "assistant").put("tool_calls", calls)
                val words = text(parts)
                if (words.isNotBlank()) assistant.put("content", words)
                messages.put(assistant)
                continue
            }
            val image = (0 until parts.length()).mapNotNull {
                parts.optJSONObject(it)?.optJSONObject("inlineData")
            }.firstOrNull()
            val role = if (content.optString("role") == "model") "assistant" else "user"
            if (image != null) {
                val mime = image.optString("mimeType")
                val data = image.optString("data")
                if (role != "user" || mime !in setOf("image/jpeg", "image/png") || data.length > 1_400_000)
                    throw IOException("The selected AI model cannot inspect this screen image.")
                messages.put(JSONObject().put("role", "user").put("content", JSONArray()
                    .put(JSONObject().put("type", "text").put("text", text(parts)))
                    .put(JSONObject().put("type", "image_url")
                        .put("image_url", JSONObject().put("url", "data:$mime;base64,$data")))))
            } else messages.put(JSONObject().put("role", role).put("content", text(parts)))
        }
        val result = JSONObject().put("model", model).put("messages", messages)
            .put("stream", true)
        // DeepSeek defaults to thinking mode, which also requires replaying
        // reasoning content across tool turns. Phone actions use its fast
        // non-thinking route and never speak or retain internal reasoning.
        if (provider == HostedTextProviders.DEEPSEEK)
            result.put("thinking", JSONObject().put("type", "disabled"))
        val config = source.optJSONObject("generationConfig")
        result.put("max_tokens", config?.optInt("maxOutputTokens", 4096) ?: 4096)
        result.put("temperature", config?.optDouble("temperature", 0.45) ?: 0.45)
        val declarations = source.optJSONArray("tools")?.optJSONObject(0)?.optJSONArray("functionDeclarations")
        if (declarations != null && declarations.length() > 0) {
            val tools = JSONArray()
            for (index in 0 until declarations.length()) {
                val declaration = declarations.optJSONObject(index) ?: continue
                tools.put(JSONObject().put("type", "function").put("function", JSONObject()
                    .put("name", declaration.optString("name"))
                    .put("description", declaration.optString("description"))
                    .put("parameters", normaliseSchema(declaration.optJSONObject("parameters") ?: JSONObject()))))
            }
            result.put("tools", tools)
            result.put("tool_choice", "auto")
        }
        return result
    }

    private fun text(parts: JSONArray): String = buildString {
        for (index in 0 until parts.length()) {
            val part = parts.optJSONObject(index) ?: continue
            if (!part.optBoolean("thought") && part.has("text")) append(part.optString("text"))
        }
    }

    private fun normaliseSchema(source: JSONObject): JSONObject {
        val result = JSONObject(source.toString())
        if (result.has("type")) result.put("type", result.optString("type").lowercase())
        val properties = result.optJSONObject("properties")
        if (properties != null) for (name in properties.keys())
            properties.optJSONObject(name)?.let { properties.put(name, normaliseSchema(it)) }
        result.optJSONObject("items")?.let { result.put("items", normaliseSchema(it)) }
        return result
    }
}

/** Bounded OpenAI-compatible stream accumulator; an action never executes here. */
internal class PhoneHostedAnswer(private val onText: (String) -> Unit) {
    private data class Tool(val id: StringBuilder = StringBuilder(),
                            val name: StringBuilder = StringBuilder(),
                            val arguments: StringBuilder = StringBuilder())
    private val tools = sortedMapOf<Int, Tool>()
    private val words = StringBuilder()
    private var finish = ""
    var received = false
        private set

    /** Token-sized events can outnumber output tokens; bound bytes instead of event count. */
    fun read(stream: InputStream, check: () -> Unit = {}, observed: () -> Unit = {}): JSONObject {
        GeminiStream.consume(stream, check = check) { event ->
            observed()
            if (event == "[DONE]") false else {
                accept(event)
                true
            }
        }
        check()
        return result()
    }

    fun accept(raw: String) {
        if (raw == "[DONE]") return
        val event = JSONObject(raw)
        if (event.has("error")) throw IOException("The hosted AI provider refused the request.")
        val choice = event.optJSONArray("choices")?.optJSONObject(0) ?: return
        received = true
        choice.optString("finish_reason").takeIf { it.isNotBlank() && it != "null" }?.let { finish = it }
        val delta = choice.optJSONObject("delta") ?: return
        if (!delta.isNull("content")) {
            val piece = delta.optString("content")
            if (words.length + piece.length > 80_000) throw IOException("The AI answer exceeded its safe length.")
            words.append(piece)
            if (piece.isNotEmpty()) onText(piece)
        }
        val calls = delta.optJSONArray("tool_calls") ?: return
        for (index in 0 until calls.length()) {
            val fragment = calls.optJSONObject(index) ?: continue
            val slot = fragment.optInt("index", index)
            if (slot !in 0..1) throw IOException("Too many phone actions were requested.")
            val tool = tools.getOrPut(slot) { Tool() }
            append(tool.id, fragment.optString("id"), 200)
            val function = fragment.optJSONObject("function") ?: continue
            append(tool.name, function.optString("name"), 120)
            append(tool.arguments, function.optString("arguments"), 20_000)
        }
    }

    fun result(): JSONObject {
        if (!received || finish.isBlank()) throw IOException("The AI answer did not finish.")
        if (finish !in setOf("stop", "tool_calls", "length"))
            throw IOException("The AI answer did not finish.")
        if (finish == "length" && tools.isNotEmpty())
            throw IOException("The AI action did not finish. It was not executed.")
        val parts = JSONArray()
        if (words.isNotEmpty()) parts.put(JSONObject().put("text", words.toString()))
        for (tool in tools.values) {
            val id = tool.id.toString()
            val name = tool.name.toString()
            val arguments = runCatching { JSONObject(tool.arguments.toString()) }.getOrNull()
            if (id.isBlank() || name.isBlank() || arguments == null)
                throw IOException("The AI returned an invalid phone action.")
            parts.put(JSONObject().put("functionCall", JSONObject().put("id", id)
                .put("name", name).put("args", arguments)))
        }
        if (parts.length() == 0) throw IOException("The AI returned an incomplete reply.")
        return JSONObject().put("candidates", JSONArray().put(JSONObject()
            .put("content", JSONObject().put("role", "model").put("parts", parts))
            .put("finishReason", if (finish == "length") "MAX_TOKENS" else "STOP")))
    }

    private fun append(out: StringBuilder, piece: String, limit: Int) {
        if (out.length + piece.length > limit) throw IOException("The AI action was too large.")
        out.append(piece)
    }
}
