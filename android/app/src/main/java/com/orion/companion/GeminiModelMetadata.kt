package com.orion.companion

import org.json.JSONObject

/** Interpret the provider's model metadata without treating a TTS model as text AI. */
internal object GeminiModelMetadata {
    fun supportsText(body: String): Boolean {
        val methods = JSONObject(body).optJSONArray("supportedGenerationMethods") ?: return false
        return (0 until methods.length()).any { methods.optString(it) == "generateContent" }
    }
}
