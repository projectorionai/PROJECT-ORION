package com.orion.companion

import android.content.Context

/** Fixed HTTPS providers for standalone phone text and reviewed phone actions. */
internal data class HostedTextConfig(val provider: String, val model: String,
                                     val endpoint: String, val key: String)

internal object HostedTextProviders {
    const val GEMINI = "gemini"
    const val GROQ = "groq"
    const val OPENROUTER = "openrouter"
    const val CLOUDFLARE = "cloudflare"
    const val CEREBRAS = "cerebras"
    const val MISTRAL = "mistral"
    const val HUGGINGFACE = "huggingface"
    const val NVIDIA = "nvidia"
    const val SAMBANOVA = "sambanova"
    const val TOGETHER = "together"
    const val DEEPSEEK = "deepseek"
    val choices = listOf(GEMINI, GROQ, OPENROUTER, CLOUDFLARE, CEREBRAS,
        MISTRAL, HUGGINGFACE, NVIDIA, SAMBANOVA, TOGETHER, DEEPSEEK)
    private val account = Regex("[a-fA-F0-9]{32}")
    private val model = Regex("[A-Za-z0-9@][A-Za-z0-9@._:/+-]{0,119}")

    fun defaultModel(provider: String): String = when (provider) {
        GROQ -> "openai/gpt-oss-20b"
        OPENROUTER -> "openai/gpt-oss-20b:free"
        CLOUDFLARE -> "@cf/google/gemma-4-26b-a4b-it"
        CEREBRAS -> "gpt-oss-120b"
        MISTRAL -> "mistral-small-latest"
        HUGGINGFACE -> "openai/gpt-oss-120b:cheapest"
        NVIDIA -> "meta/llama-3.3-70b-instruct"
        SAMBANOVA -> "Meta-Llama-3.3-70B-Instruct"
        TOGETHER -> "Qwen/Qwen3.5-9B"
        DEEPSEEK -> "deepseek-flash"
        else -> "gemini-3.5-flash-lite"
    }

    fun endpoint(provider: String, accountId: String = ""): String? = when (provider) {
        GROQ -> "https://api.groq.com/openai/v1/chat/completions"
        OPENROUTER -> "https://openrouter.ai/api/v1/chat/completions"
        CLOUDFLARE -> if (account.matches(accountId))
            "https://api.cloudflare.com/client/v4/accounts/$accountId/ai/v1/chat/completions" else null
        CEREBRAS -> "https://api.cerebras.ai/v1/chat/completions"
        MISTRAL -> "https://api.mistral.ai/v1/chat/completions"
        HUGGINGFACE -> "https://router.huggingface.co/v1/chat/completions"
        NVIDIA -> "https://integrate.api.nvidia.com/v1/chat/completions"
        SAMBANOVA -> "https://api.sambanova.ai/v1/chat/completions"
        TOGETHER -> "https://api.together.ai/v1/chat/completions"
        DEEPSEEK -> "https://api.deepseek.com/chat/completions"
        else -> null
    }

    fun label(provider: String): String = when (provider) {
        GEMINI -> "Gemini (current key)"
        GROQ -> "Groq"
        OPENROUTER -> "OpenRouter"
        CLOUDFLARE -> "Cloudflare Workers AI"
        CEREBRAS -> "Cerebras"
        MISTRAL -> "Mistral"
        HUGGINGFACE -> "Hugging Face"
        NVIDIA -> "NVIDIA NIM"
        SAMBANOVA -> "SambaNova"
        TOGETHER -> "Together AI"
        DEEPSEEK -> "DeepSeek"
        else -> "Unknown provider"
    }

    fun validModel(value: String) = model.matches(value) && !value.contains("://") &&
        !value.contains("..")

    fun validKey(value: String): Boolean = value.length in 1..512 &&
        value.all { it.code in 33..126 }
}

/** Each provider key is sealed with Android Keystore and never enters the WebView. */
internal object PhoneHostedProfile {
    private const val FILE = "orion_hosted_text"
    private fun prefs(context: Context) = context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
    fun selected(context: Context): String = prefs(context).getString("provider", HostedTextProviders.GEMINI)
        ?: HostedTextProviders.GEMINI
    fun selectGemini(context: Context) {
        prefs(context).edit().putString("provider", HostedTextProviders.GEMINI).apply()
    }
    fun model(context: Context, provider: String): String =
        prefs(context).getString("model_$provider", HostedTextProviders.defaultModel(provider))
            ?: HostedTextProviders.defaultModel(provider)
    fun accountId(context: Context): String = prefs(context).getString("cloudflare_account", "").orEmpty()
    fun hasKey(context: Context, provider: String): Boolean =
        !PhoneSecretStore.get(context, "hosted_text_$provider").isNullOrBlank()

    fun save(context: Context, provider: String, key: String, model: String,
             accountId: String = ""): Boolean {
        if (provider !in HostedTextProviders.choices || provider == HostedTextProviders.GEMINI ||
            !HostedTextProviders.validModel(model) ||
            HostedTextProviders.endpoint(provider, accountId) == null ||
            (key.isNotEmpty() && !HostedTextProviders.validKey(key))) return false
        if (key.isNotBlank() && !PhoneSecretStore.put(context, "hosted_text_$provider", key)) return false
        if (!hasKey(context, provider)) return false
        val edit = prefs(context).edit().putString("model_$provider", model)
            .putString("provider", provider)
        if (provider == HostedTextProviders.CLOUDFLARE) edit.putString("cloudflare_account", accountId)
        return edit.commit()
    }

    fun config(context: Context): HostedTextConfig? {
        val provider = selected(context)
        val model = model(context, provider)
        val endpoint = HostedTextProviders.endpoint(provider, accountId(context)) ?: return null
        val key = PhoneSecretStore.get(context, "hosted_text_$provider")?.takeIf { it.isNotBlank() }
            ?: return null
        if (!HostedTextProviders.validModel(model)) return null
        return HostedTextConfig(provider, model, endpoint, key)
    }

    fun ready(context: Context): Boolean = if (selected(context) == HostedTextProviders.GEMINI)
        PhoneAiProfile.hasKey(context) else config(context) != null

    fun forget(context: Context) {
        for (provider in HostedTextProviders.choices.filter { it != HostedTextProviders.GEMINI })
            PhoneSecretStore.remove(context, "hosted_text_$provider")
        prefs(context).edit().clear().apply()
    }
}
