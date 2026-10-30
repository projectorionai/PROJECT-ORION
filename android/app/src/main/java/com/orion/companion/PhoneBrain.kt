package com.orion.companion

import android.content.Context
import android.os.SystemClock
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.io.Closeable
import java.io.EOFException
import java.io.IOException
import java.net.ConnectException
import java.net.SocketException
import java.net.SocketTimeoutException
import java.net.UnknownHostException
import java.util.Base64
import java.util.UUID
import java.security.MessageDigest
import java.time.Instant
import java.time.ZoneId
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicLong
import javax.net.ssl.SSLException

private class GeminiHttpException(val status: Int, val retryAfterSeconds: Long?,
                                  val dailyQuota: Boolean = false) :
    IOException("Gemini returned HTTP $status.")

private class HostedHttpException(val provider: String, val status: Int,
                                  val retryAfterSeconds: Long? = null) :
    IOException("Hosted AI returned HTTP $status.")

/** Match Gemini text models to the work without mixing in Live or speech IDs. */
internal object PhoneTextModels {
    private const val LITE = "gemini-3.5-flash-lite"
    private const val BACKUP = "gemini-3.1-flash-lite"

    /** Never silently spend a customer's key on a stronger model than they selected. */
    fun candidates(preferred: String): List<String> = when (preferred.ifBlank { LITE }) {
        LITE -> listOf(LITE, BACKUP)
        BACKUP -> listOf(BACKUP, LITE)
        else -> listOf(preferred)
    }

    fun dailyQuota(detail: String): Boolean =
        Regex("(?i)per.?day|daily|requests.?per.?day|\\brpd\\b").containsMatchIn(detail)

    fun retryDelayMillis(until: List<Long>, now: Long): Long {
        val remaining = until.map { it - now }
        return if (remaining.isNotEmpty() && remaining.all { it > 0L })
            remaining.minOrNull() ?: 0L else 0L
    }

    fun pauseMillis(status: Int, retryAfterSeconds: Long?, daily: Boolean,
                    nowEpochMillis: Long): Long = when {
        status == 429 && daily -> {
            val zone = ZoneId.of("America/Los_Angeles")
            val midnight = Instant.ofEpochMilli(nowEpochMillis).atZone(zone)
                .toLocalDate().plusDays(1).atStartOfDay(zone).toInstant().toEpochMilli()
            (midnight - nowEpochMillis + 1_000L).coerceIn(60_000L, 25 * 60 * 60_000L)
        }
        status == 429 -> maxOf(60_000L, (retryAfterSeconds ?: 0L)
            .coerceIn(0L, 900L) * 1_000L)
        status == 404 -> 6 * 60 * 60_000L
        status in 500..599 -> 30_000L
        else -> 0L
    }
}

/** Direct phone AI. Actions cross a native approval boundary and return a receipt. */
class PhoneBrain(context: Context) : Closeable {
    companion object {
        private val activeRequests = java.util.concurrent.ConcurrentHashMap.newKeySet<String>()
        fun isRequestActive(id: String): Boolean = id.isNotBlank() && activeRequests.contains(id)
        /** A useful on-screen error without leaking the key, request body or provider response. */
        fun failureText(failure: Exception): String = when (failure) {
            is HostedHttpException -> when (failure.status) {
                400, 422 -> "The selected ${failure.provider} model rejected this request (HTTP ${failure.status}). Choose a chat model that supports tools in Options."
                401, 403 -> "The ${failure.provider} key was refused (HTTP ${failure.status}). Check it in Options."
                404 -> "The selected ${failure.provider} model was not found (HTTP 404). Check its model ID in Options."
                408 -> "The ${failure.provider} service timed out (HTTP 408). Please try again."
                429 -> "The ${failure.provider} request limit has been reached (HTTP 429). Wait before trying again."
                in 500..599 -> "The ${failure.provider} service is temporarily unavailable (HTTP ${failure.status})."
                else -> "The ${failure.provider} request failed (HTTP ${failure.status})."
            }
            is GeminiHttpException -> when (failure.status) {
                400, 422 -> "The AI rejected this request (HTTP ${failure.status}). Please report this code."
                401, 403 -> "The AI key was refused (HTTP ${failure.status}). Reconnect the phone to provision it again."
                408 -> "The AI service timed out (HTTP 408). Please try again."
                429 -> "The AI request limit has been reached (HTTP 429). Wait before trying again."
                in 500..599 -> "The AI service is temporarily unavailable (HTTP ${failure.status}). Please try again."
                else -> "The AI request failed (HTTP ${failure.status})."
            }
            is UnknownHostException -> "The phone cannot find the AI service. Check mobile data or Wi-Fi."
            is SocketTimeoutException -> "The AI connection timed out. Check the phone's connection and try again."
            is ConnectException, is SocketException -> "The AI connection dropped. Check mobile data or Wi-Fi."
            is SSLException -> "The secure AI connection failed. Check the phone's date and network."
            is EOFException -> "The AI reply was interrupted. Please try again."
            is IOException -> when (failure.message) {
                "Set up the phone AI key first." -> "The phone AI key is missing. Open Options to set it up."
                "Set up the selected text AI key first." -> "Set up the selected text AI provider in Options."
                "The phone request was stopped." -> "The request was stopped."
                "The AI answer did not finish.", "Gemini returned an empty response.",
                "Gemini returned no response." -> "The AI returned an incomplete reply. Please try again."
                "Speech transcription returned no answer.", "Speech transcription did not finish." ->
                    "Speech recognition returned no words. Please try again."
                "Every phone AI model is temporarily unavailable." ->
                    "The available AI models have reached their limits. Please try again later."
                else -> "The AI request failed (${failure.javaClass.simpleName}). Please try again."
            }
            else -> "The phone could not finish this request (${failure.javaClass.simpleName})."
        }
        private const val API = "https://generativelanguage.googleapis.com/v1beta/models/"
        private const val FAST_LITE_MODEL = "gemini-3.5-flash-lite"
        private const val FAST_MODEL = "gemini-3.5-flash"
        private const val BACKUP_LITE_MODEL = "gemini-3.1-flash-lite"
        private val TRANSCRIPTION_MODELS = listOf(FAST_LITE_MODEL, BACKUP_LITE_MODEL, FAST_MODEL,
            "gemini-3.6-flash")
        private const val MAX_ANSWER = 80_000
        private val ACTIONS = PhoneCommandContract.deviceActions +
            setOf("capture_dom", "click", "scroll", "set_text")
        private val FIELDS = setOf("kind", "name", "package", "number", "to", "subject", "body",
            "url", "query", "text", "screen", "hour", "minutes", "seconds", "label", "title",
            "start", "end", "location", "description", "email", "snapshotId", "nodeId", "direction")
        private val NUMERIC = setOf("hour", "minutes", "seconds", "start", "end")
    }

    private val app = context.applicationContext
    private val http = OkHttpClient.Builder().connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS).callTimeout(180, TimeUnit.SECONDS)
        .retryOnConnectionFailure(false).followRedirects(false).followSslRedirects(false).build()
    private val mcp = PhoneMcpClient(app)
    private val generation = AtomicLong()
    @Volatile private var activeCall: Call? = null
    @Volatile private var closed = false
    @Volatile private var requestId = ""

    private fun check(id: Long) {
        if (closed || generation.get() != id || Thread.currentThread().isInterrupted)
            throw IOException("The phone request was stopped.")
    }

    private fun modelPauseKey(model: String): String {
        // Only a short digest is stored. A newly provisioned key therefore
        // gets fresh limits without exposing the credential in preferences.
        val key = PhoneAiProfile.key(app).orEmpty()
        val digest = MessageDigest.getInstance("SHA-256").digest(key.toByteArray(Charsets.UTF_8))
        val account = digest.take(8).joinToString("") { "%02x".format(it.toInt() and 0xff) }
        return "$account:$model"
    }

    private fun modelReady(model: String): Boolean {
        val prefs = app.getSharedPreferences("orion_phone_ai_cooldowns", Context.MODE_PRIVATE)
        val key = modelPauseKey(model)
        val until = prefs.getLong(key, 0L)
        if (until > System.currentTimeMillis()) return false
        if (until != 0L) prefs.edit().remove(key).apply()
        return true
    }

    /** Return a delay only when every eligible model is already cooling down. */
    private fun modelsRetryDelayMillis(models: List<String>): Long {
        val now = System.currentTimeMillis()
        val prefs = app.getSharedPreferences("orion_phone_ai_cooldowns", Context.MODE_PRIVATE)
        return PhoneTextModels.retryDelayMillis(models.map { prefs.getLong(modelPauseKey(it), 0L) }, now)
    }

    fun transcriptionRetryDelayMillis(): Long = modelsRetryDelayMillis(TRANSCRIPTION_MODELS)

    fun replyRetryDelayMillis(): Long = if (PhoneHostedProfile.selected(app) == HostedTextProviders.GEMINI)
        modelsRetryDelayMillis(PhoneTextModels.candidates(PhoneAiProfile.model(app))) else 0L

    private fun pauseModel(model: String, failure: GeminiHttpException) {
        val now = System.currentTimeMillis()
        val pause = PhoneTextModels.pauseMillis(failure.status, failure.retryAfterSeconds,
            failure.dailyQuota, now)
        if (pause <= 0L) return
        app.getSharedPreferences("orion_phone_ai_cooldowns", Context.MODE_PRIVATE)
            .edit().putLong(modelPauseKey(model), now + pause).commit()
    }

    fun cancel() {
        generation.incrementAndGet()
        activeRequests.remove(requestId)
        requestId = ""
        activeCall?.cancel()
        mcp.cancel()
        PhoneActionCoordinator.cancelAll(app)
    }

    override fun close() { closed = true; cancel(); mcp.close(); closeHttpResources(http) }

    /** Speech recognition uses Gemini audio input, even whilst the host PC is off. */
    fun transcribe(wav: ByteArray): String {
        require(wav.size in 44..1_200_000) { "Voice clip is outside the supported size." }
        val id = generation.get()
        val contents = JSONArray().put(JSONObject().put("role", "user").put("parts", JSONArray()
            .put(JSONObject().put("text", "Transcribe the spoken words exactly in their original language. Return only the words. If there is no intelligible speech, return an empty response."))
            .put(JSONObject().put("inlineData", JSONObject().put("mimeType", "audio/wav")
                .put("data", Base64.getEncoder().encodeToString(wav))))))
        val payload = JSONObject().put("contents", contents)
            .put("generationConfig", JSONObject().put("temperature", 0).put("maxOutputTokens", 1024))
        var events: List<JSONObject>? = null
        var lastFailure: GeminiHttpException? = null
        // Speech and replies draw from separate model limits. A model that has
        // just refused a request is skipped briefly across both call paths.
        for (model in TRANSCRIPTION_MODELS) {
            if (!modelReady(model)) continue
            try {
                events = generate(payload, id, stream = false, modelOverride = model)
                break
            } catch (failure: GeminiHttpException) {
                if (failure.status !in setOf(400, 404, 429, 503)) throw failure
                pauseModel(model, failure)
                lastFailure = failure
            }
        }
        val completed = events ?: throw (lastFailure ?: IOException("Every phone AI model is temporarily unavailable."))
        val candidate = completed.lastOrNull()?.optJSONArray("candidates")?.optJSONObject(0)
            ?: throw IOException("Speech transcription returned no answer.")
        if (candidate.optString("finishReason") != "STOP") throw IOException("Speech transcription did not finish.")
        val parts = candidate.optJSONObject("content")?.optJSONArray("parts") ?: JSONArray()
        return (0 until parts.length()).joinToString("") { parts.optJSONObject(it)?.optString("text") ?: "" }.trim().take(16_000)
    }

    fun reply(message: String, onDelta: (String) -> Unit = {},
              execute: (JSONObject, (JSONObject) -> Unit) -> Unit): String {
        require(message.isNotBlank() && message.length <= 16_000)
        // A delayed stop from an older reply cannot cancel this new turn's actions before speech begins.
        SpeechPlaybackEpoch.begin()
        val startedAt = PhonePerformance.now()
        var firstTextSeen = false
        var modelRequests = 0
        var selectedModel = ""
        fun textArrived(piece: String) {
            if (piece.isEmpty()) return
            if (!firstTextSeen) {
                firstTextSeen = true
                PhonePerformance.record(app, "first_text", startedAt)
            }
            onDelta(piece)
        }
        val id = generation.get()
        val request = UUID.randomUUID().toString()
        requestId = request
        activeRequests.add(request)
        try {
            val hosted = if (PhoneHostedProfile.selected(app) == HostedTextProviders.GEMINI) null
                else PhoneHostedProfile.config(app) ?: throw IOException("Set up the selected text AI key first.")
            val contents = PhoneConversationStore.latest(app)
            PhonePerformance.context(app, (0 until contents.length()).sumOf { index ->
                contents.optJSONObject(index)?.optJSONArray("parts")?.optJSONObject(0)?.optString("text")?.length ?: 0
            })
            if (contents.length() == 0 || contents.getJSONObject(contents.length() - 1)
                    .optJSONArray("parts")?.optJSONObject(0)?.optString("text") != message)
                contents.put(JSONObject().put("role", "user").put("parts", JSONArray().put(JSONObject().put("text", message))))
            if (Regex("\\b(?:screen|screenshot|display|see|look|visible|read this)\\b", RegexOption.IGNORE_CASE)
                    .containsMatchIn(message)) {
                OrionProjectionService.latestFrame()?.let { frame ->
                    contents.getJSONObject(contents.length() - 1).getJSONArray("parts")
                        .put(JSONObject().put("inlineData", JSONObject().put("mimeType", "image/jpeg")
                            .put("data", Base64.getEncoder().encodeToString(frame))))
                }
            }
            val available = if (PhoneMcpClient.configured(app)) mcp.discover() else null
            val mcpNames = if (available != null) (0 until minOf(available.length(), 100)).joinToString("\n") {
                val item = available.optJSONObject(it)
                "${item?.optString("name")}: ${item?.optString("description")?.take(180)}"
            }.take(12_000) else "No MCP server is connected."
            val system = "You are ORION on the user's Android phone. Answer in British English. " +
                "The phone works independently of the PC. A request without an explicit PC target is for this phone. " +
                "Never claim an action ran until its function receipt confirms it. Never invent an e-mail recipient, " +
                "message body, app identity or time. Ask one precise follow-up when details are missing. " +
                "Use phone_action only for an actual phone action and only with fields the action requires. " +
                "Actions use native review; permitted app and offline-map launches can run hands-free. " +
                "No arbitrary Android intents or hidden tool execution. " +
                PcPowerPresentation.assistantScope +
                "For car directions use the navigate phone action. An address opens OsmAnd's offline " +
                "destination search. A supplied precise coordinate opens ORION's native offline route screen, " +
                "which needs BRouter, downloaded routing data and a fresh GPS fix. " +
                "Never invent a destination's coordinates. The map-screen launch is not proof that a route " +
                "was calculated or navigation started. Spoken and locked-screen turn guidance still uses OsmAnd. " +
                "Tell the user to check the route and download every required region before driving. " +
                "To open an app, call open_app with the natural name the user said; a package is unnecessary. " +
                "For 'open a game', pass name 'game'. Android resolves a unique installed game or returns " +
                "choices. If several apps match, ask which one; do not claim one opened. " +
                "For current popular YouTube videos use show_trending_videos. " +
                "For this phone's battery, charging or connection state use device_status; never substitute PC telemetry. " +
                "For long questions give a complete answer; continue until the answer is finished. " +
                "MCP tools on this phone (only if listed):\n$mcpNames"
            val tools = JSONArray().put(JSONObject().put("functionDeclarations", declarations(available)))
            PhonePerformance.record(app, "preparation", startedAt)
            val answer = StringBuilder()
            var toolCalls = 0
            var continuations = 0
            var replyModel: String? = null
            val actionLike = Regex("^(?:please\\s+)?(?:open|launch|play|send|e-?mail|call|text|write|create|set|tap|click|scroll|share|navigate|find|list|show|start|stop|run|read|move|delete)\\b",
                RegexOption.IGNORE_CASE).containsMatchIn(message.trim())
            while (true) {
                check(id)
                val payload = JSONObject().put("systemInstruction", JSONObject().put("parts", JSONArray()
                    .put(JSONObject().put("text", system))))
                    .put("contents", contents).put("tools", tools)
                    .put("generationConfig", JSONObject().put("temperature", 0.45).put("maxOutputTokens", 4096))
                val emitLive = !actionLike || toolCalls > 0
                val deliver: (String) -> Unit = { piece ->
                    if (emitLive) textArrived(piece)
                }
                val events = if (hosted != null) {
                    modelRequests++
                    selectedModel = hosted.model
                    listOf(generateHosted(payload, hosted, id, deliver))
                } else if (replyModel != null) {
                    // Keep the model that produced the function call or partial
                    // answer; repeating either on another model is unsafe.
                    val pinnedModel = replyModel
                    modelRequests++
                    try {
                        generate(payload, id, stream = true, modelOverride = pinnedModel,
                            retryAllowed = toolCalls == 0, onText = deliver)
                    } catch (failure: GeminiHttpException) {
                        pauseModel(pinnedModel, failure)
                        throw failure
                    }
                } else {
                    val candidates = PhoneTextModels.candidates(PhoneAiProfile.model(app))
                    var firstEvents: List<JSONObject>? = null
                    var lastFailure: GeminiHttpException? = null
                    for (model in candidates) {
                        if (!modelReady(model)) continue
                        try {
                            modelRequests++
                            firstEvents = generate(payload, id, stream = true, modelOverride = model,
                                retryAllowed = true, onText = deliver)
                            replyModel = model
                            selectedModel = model
                            break
                        } catch (failure: GeminiHttpException) {
                            // HTTP failure arrives before the first stream event.
                            // No native action or text has been committed yet.
                            if (failure.status !in setOf(404, 429, 500, 502, 503, 504) || toolCalls != 0 ||
                                continuations != 0 || answer.isNotEmpty()) throw failure
                            pauseModel(model, failure)
                            lastFailure = failure
                        }
                    }
                    firstEvents ?: throw (lastFailure ?: IOException("Every phone AI model is temporarily unavailable."))
                }
                val modelParts = JSONArray()
                val calls = mutableListOf<JSONObject>()
                val turnText = StringBuilder()
                var reason = ""
                for (event in events) {
                    val candidate = event.optJSONArray("candidates")?.optJSONObject(0) ?: continue
                    val parts = candidate.optJSONObject("content")?.optJSONArray("parts") ?: JSONArray()
                    for (index in 0 until parts.length()) {
                        val part = parts.optJSONObject(index) ?: continue
                        if (part.has("functionCall")) {
                            val call = part.getJSONObject("functionCall")
                            calls.add(call)
                            // Gemini 3 tool turns may carry a thought signature. Preserve
                            // the complete part when returning the model turn to Gemini.
                            modelParts.put(JSONObject(part.toString()))
                        } else if (part.has("text")) {
                            val chunk = part.optString("text")
                            modelParts.put(JSONObject(part.toString()))
                            if (!part.optBoolean("thought")) turnText.append(chunk)
                        }
                    }
                    val current = candidate.optString("finishReason")
                    if (current.isNotEmpty()) reason = current
                }
                check(id)
                if (calls.isNotEmpty()) {
                    if (++toolCalls > 6 || calls.size > 2) throw IOException("Too many phone actions were requested.")
                    contents.put(JSONObject().put("role", "model").put("parts", modelParts))
                    val responses = JSONArray()
                    for (call in calls) {
                        check(id)
                        val name = call.optString("name")
                        val args = call.optJSONObject("args") ?: JSONObject()
                        val receipt = when (name) {
                            "phone_action" -> runAction(args, request, id, message, execute)
                            "phone_mcp" -> runMcp(args, request, id, execute, available)
                            else -> JSONObject().put("ok", false).put("detail", "Unrecognised phone function.")
                        }
                        val response = JSONObject().put("name", name).put("response", receipt)
                        call.optString("id").takeIf { it.isNotBlank() }?.let { response.put("id", it) }
                        responses.put(JSONObject().put("functionResponse", response))
                    }
                    // Gemini 3 requires the results of a parallel function-call
                    // turn together as the next user turn. The old "function"
                    // role now receives HTTP 400 from GenerateContent.
                    contents.put(JSONObject().put("role", "user").put("parts", responses))
                    continue
                }
                if (reason !in setOf("STOP", "MAX_TOKENS")) throw IOException("The AI answer did not finish.")
                val section = turnText.toString()
                if (section.isNotEmpty()) {
                    if (answer.length + section.length > MAX_ANSWER) throw IOException("The AI answer exceeded its safe length.")
                    answer.append(section)
                    if (!emitLive) textArrived(section)
                }
                if (reason == "STOP") return answer.toString().trim().ifEmpty { "I have no verified result yet. Please try again." }
                if (++continuations >= 5) return answer.toString().trimEnd() +
                    "\n\nI have more to cover. Say ‘continue’ and I will carry on from here."
                contents.put(JSONObject().put("role", "model").put("parts", JSONArray().put(JSONObject().put("text", section))))
                contents.put(JSONObject().put("role", "user").put("parts", JSONArray().put(JSONObject()
                    .put("text", "Continue the previous answer exactly where it stopped. Do not repeat earlier text."))))
            }
        } finally {
            PhonePerformance.record(app, "reply", startedAt)
            if (selectedModel.isNotEmpty()) PhonePerformance.model(app, selectedModel, modelRequests)
            activeRequests.remove(request)
            if (requestId == request) requestId = ""
        }
    }

    private fun declarations(mcpTools: JSONArray?): JSONArray {
        val properties = JSONObject().put("kind", JSONObject().put("type", "STRING")
            .put("enum", JSONArray(ACTIONS.sorted())))
        for (field in FIELDS - "kind") {
            val property = JSONObject().put("type", if (field in NUMERIC) "INTEGER" else "STRING")
            if (field == "name") property.put("description", "For open_app, pass the app's natural name or a category such as game. Do not require a package name.")
            if (field == "package") property.put("description", "Optional installed app package; normally omit for open_app.")
            properties.put(field, property)
        }
        val phone = JSONObject().put("name", "phone_action")
            .put("description", "Run one reviewed native action on this Android phone. For open_app, use the user's natural app name or category; other action details must be user supplied.")
            .put("parameters", JSONObject().put("type", "OBJECT").put("properties", properties)
                .put("required", JSONArray().put("kind")))
        val result = JSONArray().put(phone)
        if (mcpTools != null && mcpTools.length() > 0) result.put(JSONObject().put("name", "phone_mcp")
            .put("description", "Call one discovered HTTPS MCP tool after exact native review.")
            .put("parameters", JSONObject().put("type", "OBJECT").put("properties", JSONObject()
                .put("tool", JSONObject().put("type", "STRING"))
                .put("arguments_json", JSONObject().put("type", "STRING")))
                .put("required", JSONArray().put("tool").put("arguments_json"))))
        return result
    }

    private fun runAction(args: JSONObject, request: String, id: Long, userMessage: String,
                          execute: (JSONObject, (JSONObject) -> Unit) -> Unit): JSONObject {
        val kind = args.optString("kind")
        if (kind !in ACTIONS || args.keys().asSequence().any { it !in FIELDS } || args.toString().length > 20_000)
            return JSONObject().put("ok", false).put("detail", "The proposed phone action was invalid.")
        if (kind in setOf("send_email", "compose_email")) {
            val recipient = args.optString("to").trim()
            val subject = args.optString("subject")
            val body = args.optString("body")
            if (recipient.isEmpty() || subject.length > 200 || body.length > 16_000 ||
                (kind == "send_email" && (body.isBlank() ||
                    (subject.isBlank() && !Regex("\\b(?:no subject|without a subject)\\b", RegexOption.IGNORE_CASE)
                        .containsMatchIn(userMessage)))))
                return JSONObject().put("ok", false).put("detail", "Give an exact recipient, subject and message before sending from the phone.")
        }
        val safe = JSONObject(args.toString()).put("_phone_brain_request", request)
            .put("_phone_brain_deadline", SystemClock.elapsedRealtime() + 120_000)
        return review(safe, id, execute)
    }

    private fun runMcp(args: JSONObject, request: String, id: Long,
                       execute: (JSONObject, (JSONObject) -> Unit) -> Unit, available: JSONArray?): JSONObject {
        val tool = args.optString("tool")
        if (available == null || (0 until available.length()).none { available.optJSONObject(it)?.optString("name") == tool })
            return JSONObject().put("ok", false).put("detail", "That MCP tool is unavailable on this phone.")
        val arguments = runCatching { JSONObject(args.optString("arguments_json")) }.getOrNull()
            ?: return JSONObject().put("ok", false).put("detail", "MCP arguments were invalid.")
        val revision = mcp.configurationId()
        val action = JSONObject().put("kind", "mcp_authorise").put("server", mcp.serverLabel())
            .put("tool", tool).put("arguments", arguments)
            .put("_phone_brain_request", request).put("_phone_brain_deadline", SystemClock.elapsedRealtime() + 120_000)
        val approval = review(action, id, execute)
        if (!approval.optBoolean("ok") || !approval.optBoolean("authorised")) return approval
        check(id)
        return try { mcp.call(tool, arguments, revision) }
        catch (_: Exception) { JSONObject().put("ok", false).put("detail", "The MCP call did not return a verified result. It was not repeated.") }
    }

    private fun review(action: JSONObject, id: Long,
                       execute: (JSONObject, (JSONObject) -> Unit) -> Unit): JSONObject {
        check(id)
        val latch = CountDownLatch(1)
        var result: JSONObject? = null
        execute(action) { receipt -> result = JSONObject(receipt.toString()); latch.countDown() }
        val end = SystemClock.elapsedRealtime() + 120_000
        while (SystemClock.elapsedRealtime() < end) {
            check(id)
            if (latch.await(200, TimeUnit.MILLISECONDS)) return result ?: JSONObject().put("ok", false)
        }
        throw IOException("Phone approval timed out. The action was cancelled.")
    }

    private fun generate(payload: JSONObject, id: Long, stream: Boolean,
                         modelOverride: String? = null, retryAllowed: Boolean = true,
                         onText: (String) -> Unit = {}): List<JSONObject> {
        check(id)
        val key = PhoneAiProfile.key(app) ?: throw IOException("Set up the phone AI key first.")
        val model = modelOverride ?: PhoneAiProfile.model(app)
        require(Regex("gemini-[A-Za-z0-9._-]{1,100}").matches(model))
        val path = if (stream) ":streamGenerateContent?alt=sse" else ":generateContent"
        val request = Request.Builder().url(API + model + path).header("x-goog-api-key", key)
            .header("Accept", if (stream) "text/event-stream" else "application/json")
            .post(payload.toString().toRequestBody("application/json; charset=utf-8".toMediaType())).build()
        var retries = 0
        while (true) {
            check(id)
            val call = http.newCall(request)
            activeCall = call
            var receivedOutput = false
            try {
                return call.execute().use { response ->
                    if (!response.isSuccessful) {
                        val detail = response.peekBody(16_384).string()
                        throw GeminiHttpException(response.code,
                            response.header("Retry-After")?.toLongOrNull(),
                            response.code == 429 && PhoneTextModels.dailyQuota(detail))
                    }
                    val body = response.body ?: throw IOException("Gemini returned no response.")
                    val output = ArrayList<JSONObject>()
                    if (stream) GeminiStream.consume(body.byteStream(), check = { check(id) }) { raw ->
                        receivedOutput = true
                        if (raw != "[DONE]") {
                            val event = JSONObject(raw)
                            output.add(event)
                            val parts = event.optJSONArray("candidates")?.optJSONObject(0)
                                ?.optJSONObject("content")?.optJSONArray("parts") ?: JSONArray()
                            if ((0 until parts.length()).none { parts.optJSONObject(it)?.has("functionCall") == true })
                                for (index in 0 until parts.length())
                                    parts.optJSONObject(index)?.takeIf { !it.optBoolean("thought") }
                                        ?.optString("text")?.takeIf { it.isNotEmpty() }?.let(onText)
                        }
                        true
                    } else {
                        val source = body.source()
                        if (source.request(2_097_153) || source.buffer.size > 2_097_152)
                            throw IOException("Gemini response was too large.")
                        receivedOutput = true
                        output.add(JSONObject(source.readUtf8()))
                    }
                    check(id)
                    if (output.isEmpty()) throw IOException("Gemini returned an empty response.")
                    output
                }
            } catch (failure: IOException) {
                check(id)
                val transient = when (failure) {
                    is GeminiHttpException -> when (failure.status) {
                        // An absent or long Retry-After usually means a quota limit. More
                        // immediate attempts would use up calls and delay the answer.
                        429 -> !failure.dailyQuota &&
                            (failure.retryAfterSeconds?.let { it in 0L..3L } ?: false)
                        500, 502, 503, 504 -> true
                        else -> false
                    }
                    is SocketTimeoutException, is ConnectException, is SocketException,
                    is UnknownHostException, is EOFException -> true
                    else -> false
                }
                if (!retryAllowed || receivedOutput || !transient || retries >= 2) throw failure
                retries++
                val normalDelay = if (retries == 1) 350L else 900L
                val serverDelay = (failure as? GeminiHttpException)?.retryAfterSeconds
                    ?.coerceIn(0L, 3L)?.times(1_000L) ?: 0L
                val deadline = SystemClock.elapsedRealtime() + maxOf(normalDelay, serverDelay)
                while (SystemClock.elapsedRealtime() < deadline) {
                    check(id)
                    try { Thread.sleep(minOf(100L, deadline - SystemClock.elapsedRealtime())) }
                    catch (_: InterruptedException) { Thread.currentThread().interrupt(); check(id) }
                }
            } finally { if (activeCall === call) activeCall = null }
        }
    }

    private fun generateHosted(payload: JSONObject, provider: HostedTextConfig,
                               id: Long, onText: (String) -> Unit): JSONObject {
        check(id)
        val requestBody = PhoneHostedChat.request(payload, provider.model, provider.provider)
        val request = Request.Builder().url(provider.endpoint)
            .header("Authorization", "Bearer ${provider.key}")
            .header("Accept", "text/event-stream")
            .post(requestBody.toString().toRequestBody("application/json; charset=utf-8".toMediaType()))
            .build()
        var retries = 0
        while (true) {
            check(id)
            val call = http.newCall(request)
            activeCall = call
            var receivedOutput = false
            try {
                return call.execute().use { response ->
                    check(id)
                    if (!response.isSuccessful) throw HostedHttpException(provider.provider,
                        response.code, response.header("Retry-After")?.toLongOrNull())
                    val body = response.body ?: throw IOException("The AI answer did not finish.")
                    val answer = PhoneHostedAnswer(onText)
                    answer.read(body.byteStream(), check = { check(id) },
                        observed = { receivedOutput = true })
                }
            } catch (failure: IOException) {
                check(id)
                val transient = when (failure) {
                    is HostedHttpException -> when (failure.status) {
                        408, 500, 502, 503, 504 -> true
                        429 -> failure.retryAfterSeconds?.let { it in 0L..3L } == true
                        else -> false
                    }
                    is SocketTimeoutException, is ConnectException, is SocketException,
                    is UnknownHostException, is EOFException -> true
                    else -> false
                }
                if (receivedOutput || !transient || retries >= 2) throw failure
                retries++
                val normalDelay = if (retries == 1) 350L else 900L
                val serverDelay = (failure as? HostedHttpException)?.retryAfterSeconds
                    ?.coerceIn(0L, 3L)?.times(1_000L) ?: 0L
                val deadline = SystemClock.elapsedRealtime() + maxOf(normalDelay, serverDelay)
                while (SystemClock.elapsedRealtime() < deadline) {
                    check(id)
                    try { Thread.sleep(minOf(100L, deadline - SystemClock.elapsedRealtime())) }
                    catch (_: InterruptedException) { Thread.currentThread().interrupt(); check(id) }
                }
            } finally { if (activeCall === call) activeCall = null }
        }
    }
}
