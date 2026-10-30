package com.orion.companion

import android.content.Context
import android.os.Handler
import android.os.Looper
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Standalone phone settings. A provider key is never sent into a WebView. */
object PhoneAiProfile {
    private val worker = Executors.newSingleThreadExecutor { r -> Thread(r, "orion-phone-provision").apply { isDaemon = true } }
    private val main = Handler(Looper.getMainLooper())
    private fun prefs(context: Context) = context.getSharedPreferences("orion_phone_ai", Context.MODE_PRIVATE)
    private fun validTextModel(value: String): Boolean =
        Regex("gemini-[A-Za-z0-9._-]{1,100}").matches(value) &&
            listOf("tts", "live", "transcribe", "embedding", "image").none {
                value.contains(it, ignoreCase = true)
            }
    fun isPhoneMode(context: Context): Boolean = prefs(context).getBoolean("phone_mode", true)
    fun setPhoneMode(context: Context, enabled: Boolean) { prefs(context).edit().putBoolean("phone_mode", enabled).apply() }
    fun key(context: Context): String? {
        val encrypted = prefs(context).getString("sealed_key", null) ?: return null
        val value = PhoneSecretStore.decrypt(encrypted)
        if (value.isNullOrBlank()) prefs(context).edit().remove("sealed_key").apply()
        return value?.takeIf { it.isNotBlank() }
    }
    fun hasKey(context: Context) = key(context) != null
    fun model(context: Context): String {
        val saved = prefs(context).getString("model", null)
        // Earlier provisioning shipped this exact default; move existing phones
        // away from its exhausted daily allowance without changing other choices.
        return if (saved.isNullOrBlank() || (saved == "gemini-2.5-flash" &&
                !prefs(context).getBoolean("explicit_text_model", false)))
            "gemini-3.5-flash-lite" else saved
    }
    fun selectTextModel(context: Context, selected: String): Boolean {
        val value = selected.trim()
        if (!validTextModel(value)) return false
        return prefs(context).edit().putString("model", value)
            .putBoolean("explicit_text_model", true).commit()
    }
    /** Check key access and text generation support before changing a working setting. */
    fun verifyTextModel(context: Context, proposedKey: String, proposedModel: String,
                        onComplete: (String?) -> Unit) {
        val key = proposedKey.trim()
        val model = proposedModel.trim()
        if (key.length !in 1..512 || key.any { it.code !in 33..126 } || !validTextModel(model)) {
            main.post { onComplete("Enter a Gemini key and a text model ID.") }
            return
        }
        worker.execute {
            val client = OkHttpClient.Builder().connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(6, TimeUnit.SECONDS).callTimeout(12, TimeUnit.SECONDS)
                .followRedirects(false).followSslRedirects(false).build()
            val problem = try {
                val request = Request.Builder()
                    .url("https://generativelanguage.googleapis.com/v1beta/models/$model")
                    .header("x-goog-api-key", key).get().build()
                client.newCall(request).execute().use { response ->
                    when (response.code) {
                        200 -> {
                            val source = response.body?.source()
                            if (source == null || source.request(16_385) || source.buffer.size > 16_384)
                                "The model check returned an invalid response."
                            else {
                                if (GeminiModelMetadata.supportsText(source.readUtf8())) null
                                else "This model cannot generate phone text replies."
                            }
                        }
                        400, 401, 403 -> "Gemini refused this key or project. Check the key in AI Studio."
                        404 -> "This text model is not available for this key."
                        429 -> "Gemini is rate limited. Try the model check later."
                        else -> "Gemini could not check the key or model (HTTP ${response.code})."
                    }
                }
            } catch (_: Exception) { "The phone could not reach Gemini to check this key and model." }
            finally { closeHttpResources(client) }
            main.post { onComplete(problem) }
        }
    }
    fun voiceModel(context: Context): String = prefs(context).getString("voice_model", "gemini-3.8-flash-tts") ?: "gemini-3.8-flash-tts"
    fun voiceName(context: Context): String = PhoneSecretStore.get(context, "voice_name")
        ?.takeIf(PhoneVoices::valid) ?: PhoneVoices.DEFAULT
    fun selectVoice(context: Context, name: String): Boolean = PhoneVoices.valid(name) &&
        PhoneSecretStore.put(context, "voice_name", name)
    fun liveVoiceEnabled(context: Context): Boolean = prefs(context).getBoolean("live_voice", true)
    fun setLiveVoiceEnabled(context: Context, enabled: Boolean) {
        prefs(context).edit().putBoolean("live_voice", enabled).apply()
    }
    fun save(context: Context, key: String, model: String = "gemini-3.5-flash-lite", voice: String = "gemini-3.8-flash-tts"): Boolean {
        if (key.length !in 1..512 || key.any { it.code !in 33..126 } ||
            !Regex("[A-Za-z0-9._-]{1,120}").matches(model) ||
            !Regex("[A-Za-z0-9._-]{1,120}").matches(voice)) return false
        val sealed = PhoneSecretStore.encrypt(key) ?: return false
        return prefs(context).edit().putString("sealed_key", sealed).putString("model", model)
            .putString("voice_model", voice).putBoolean("phone_mode", true).commit()
    }
    fun forget(context: Context) { prefs(context).edit().remove("sealed_key")
        .remove("model").remove("explicit_text_model").apply() }
    fun provision(context: Context, onComplete: (String?) -> Unit) {
        val app = context.applicationContext
        worker.execute {
            var failure = "Connect to the PC once to set up ORION on this phone, or enter an AI key."
            var succeeded = false
            val credentials = Prefs.credentials(app)
            if (credentials != null) {
                val client = pinnedHttpClient(app).newBuilder().connectTimeout(6, TimeUnit.SECONDS)
                    .readTimeout(12, TimeUnit.SECONDS).callTimeout(18, TimeUnit.SECONDS).build()
                try {
                    // Learned HTTPS routes are ordered for the current network;
                    // the manual address is already the final fallback.
                    for (endpoint in Prefs.candidateUrls(app).take(4)) {
                        val base = endpoint.toHttpUrlOrNull() ?: continue
                        if (base.scheme != "https") continue
                        try {
                            val tokenBody = JSONObject().put("device_id", credentials.first).put("refresh_token", credentials.second)
                                .toString().toRequestBody("application/json".toMediaType())
                            val token = client.newCall(Request.Builder().url(base.newBuilder().encodedPath("/v1/auth/token").query(null).build())
                                .post(tokenBody).build()).execute().use { response ->
                                if (!response.isSuccessful) null else JSONObject(response.body?.string().orEmpty()).optString("access_token").takeIf { it.isNotBlank() }
                            } ?: continue
                            client.newCall(Request.Builder().url(base.newBuilder().encodedPath("/v1/device/ai-profile").query(null).build())
                                .header("Authorization", "Bearer $token").post("{}".toRequestBody("application/json".toMediaType())).build())
                                .execute().use { response ->
                                    val source = response.body?.source() ?: return@use
                                    if (source.request(16385) || source.buffer.size > 16384) { failure = "AI provisioning returned an invalid profile."; return@use }
                                    val result = JSONObject(source.readUtf8())
                                    if (!response.isSuccessful || !result.optBoolean("ok")) {
                                        failure = result.optString("error", "AI provisioning was refused.").take(200)
                                    } else if (result.optBoolean("requires_encrypted_storage")) {
                                        succeeded = save(app, result.optString("api_key"), result.optString("model"), result.optString("voice_model"))
                                        if (!succeeded) failure = "Android Keystore could not seal the AI key. Nothing was stored."
                                    }
                                }
                            if (succeeded) break
                        } catch (_: Exception) { /* Try another previously paired HTTPS endpoint. */ }
                    }
                } finally { closeHttpResources(client) }
            }
            main.post { onComplete(if (succeeded) null else failure) }
        }
    }
}
