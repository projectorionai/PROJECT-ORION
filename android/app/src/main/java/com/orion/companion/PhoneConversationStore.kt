package com.orion.companion

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/** Bounded, encrypted phone memory, available without a desktop connection. */
object PhoneConversationStore {
    @Synchronized fun entries(context: Context): JSONArray {
        return runCatching { JSONArray(PhoneSecretStore.get(context, "phone_conversation") ?: "[]") }.getOrDefault(JSONArray())
    }
    @Synchronized fun append(context: Context, role: String, text: String) {
        val old = entries(context)
        val next = old.put(JSONObject().put("id", UUID.randomUUID().toString())
            .put("role", if (role == "user") "user" else "model").put("text", text.take(65536))
            .put("at", System.currentTimeMillis()))
        var count = next.length()
        var size = (0 until count).sumOf { next.getJSONObject(it).optString("text").length }
        var first = 0
        while (count > 300 || size > 1_048_576) {
            size -= next.getJSONObject(first++).optString("text").length
            count--
        }
        val bounded = JSONArray()
        for (i in first until next.length()) bounded.put(next.getJSONObject(i))
        check(PhoneSecretStore.put(context, "phone_conversation", bounded.toString())) { "The phone could not save its encrypted conversation." }
    }
    fun latest(context: Context): JSONArray {
        val old = entries(context)
        val result = JSONArray()
        val history = (0 until old.length()).map { index ->
            old.getJSONObject(index).let { PhonePromptLine(it.optString("role"), it.optString("text")) }
        }
        PhonePromptWindow.select(history).forEach { row ->
            result.put(JSONObject().put("role", row.role)
                .put("parts", JSONArray().put(JSONObject().put("text", row.text))))
        }
        return result
    }
    fun clear(context: Context) { PhoneSecretStore.remove(context, "phone_conversation") }
}
