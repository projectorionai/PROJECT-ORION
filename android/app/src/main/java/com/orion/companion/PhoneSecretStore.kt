package com.orion.companion

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/** A separate Keystore key for phone AI/MCP credentials; unpairing cannot erase it. */
object PhoneSecretStore {
    private const val ALIAS = "orion_phone_ai_v1"
    fun get(context: android.content.Context, name: String): String? {
        val stored = context.getSharedPreferences("orion_phone_secrets", android.content.Context.MODE_PRIVATE).getString(name, null) ?: return null
        return decrypt(stored)
    }
    fun put(context: android.content.Context, name: String, value: String): Boolean {
        val sealed = encrypt(value) ?: return false
        return context.getSharedPreferences("orion_phone_secrets", android.content.Context.MODE_PRIVATE).edit().putString(name, sealed).commit()
    }
    fun remove(context: android.content.Context, name: String) {
        context.getSharedPreferences("orion_phone_secrets", android.content.Context.MODE_PRIVATE).edit().remove(name).apply()
    }
    private fun key(create: Boolean): SecretKey? {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        (store.getEntry(ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        if (!create) return null
        return KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore").apply {
            init(KeyGenParameterSpec.Builder(ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT)
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM).setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256).build())
        }.generateKey()
    }
    fun encrypt(value: String): String? = runCatching {
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, key(true))
        "v1:" + Base64.encodeToString(cipher.iv, Base64.NO_WRAP) + ":" +
            Base64.encodeToString(cipher.doFinal(value.toByteArray(Charsets.UTF_8)), Base64.NO_WRAP)
    }.getOrNull()
    fun decrypt(value: String): String? = runCatching {
        require(value.startsWith("v1:"))
        val parts = value.removePrefix("v1:").split(':')
        require(parts.size == 2)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, key(false) ?: error("Key unavailable"),
            GCMParameterSpec(128, Base64.decode(parts[0], Base64.NO_WRAP)))
        String(cipher.doFinal(Base64.decode(parts[1], Base64.NO_WRAP)), Charsets.UTF_8)
    }.getOrNull()
    fun deleteKey() { runCatching { KeyStore.getInstance("AndroidKeyStore").apply { load(null) }.deleteEntry(ALIAS) } }
}
