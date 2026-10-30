package com.orion.companion

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Encrypts the pairing's refresh credential with a key that never leaves the
 * phone's Android Keystore.
 *
 * The refresh credential is the long-lived half of pairing: whoever holds it
 * can mint access tokens for ORION. It used to sit in ordinary private
 * preferences, readable by anything that can read the app's files (a rooted
 * phone, a debugging bridge, a careless backup tool). Stored as AES-GCM
 * ciphertext under a Keystore key, the file alone is useless.
 *
 * The key is created without user-authentication requirements on purpose:
 * the credential is needed by background services (the sentinel, the uplink)
 * while the screen is locked. That is protection against copying the data,
 * not against someone holding the unlocked phone — which the pairing itself
 * can be revoked for from the desktop.
 */
object SecureStore {
    private const val TAG = "OrionSecureStore"
    private const val KEYSTORE = "AndroidKeyStore"
    private const val ALIAS = "orion_refresh_credential_v1"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val PREFIX = "v1:"
    private const val TAG_BITS = 128

    private fun key(create: Boolean): SecretKey? {
        val store = KeyStore.getInstance(KEYSTORE).apply { load(null) }
        (store.getEntry(ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        if (!create) return null
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, KEYSTORE)
        generator.init(
            KeyGenParameterSpec.Builder(
                ALIAS, KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    /** Ciphertext for [plain], or null if the Keystore refused (the caller
     *  must then NOT fall back to storing the plain value). */
    fun encrypt(plain: String): String? = try {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key(create = true))
        val sealed = cipher.doFinal(plain.toByteArray(Charsets.UTF_8))
        PREFIX + Base64.encodeToString(cipher.iv, Base64.NO_WRAP) + ":" +
            Base64.encodeToString(sealed, Base64.NO_WRAP)
    } catch (e: Exception) {
        Log.w(TAG, "could not encrypt the pairing credential: ${e.javaClass.simpleName}")
        null
    }

    /**
     * The plain value, or null when it cannot be recovered — the key was
     * wiped (device reset of the Keystore, a backup restored onto another
     * phone) or the stored text was altered. Null means "pair again", never
     * "carry on with something half-read".
     */
    fun decrypt(stored: String): String? {
        if (!stored.startsWith(PREFIX)) return null
        return try {
            val body = stored.removePrefix(PREFIX)
            val iv = Base64.decode(body.substringBefore(':'), Base64.NO_WRAP)
            val sealed = Base64.decode(body.substringAfter(':'), Base64.NO_WRAP)
            val secret = key(create = false) ?: return null
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(Cipher.DECRYPT_MODE, secret, GCMParameterSpec(TAG_BITS, iv))
            String(cipher.doFinal(sealed), Charsets.UTF_8)
        } catch (e: Exception) {
            Log.w(TAG, "the stored pairing credential cannot be read: ${e.javaClass.simpleName}")
            null
        }
    }

    /** Forget the key too (a deliberate unpair). */
    fun deleteKey() {
        try {
            KeyStore.getInstance(KEYSTORE).apply { load(null) }.deleteEntry(ALIAS)
        } catch (_: Exception) {
        }
    }
}
