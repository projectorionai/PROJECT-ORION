package com.orion.companion

import android.content.Context
import android.net.Uri
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import org.json.JSONArray
import org.json.JSONObject

/**
 * Where the phone remembers how to reach ORION.
 *
 * Beyond the manually entered host, it stores the HTTPS endpoints reported at
 * pairing time. Serve mode supplies the MagicDNS name on port 443; raw
 * Tailscale IPs and cleartext endpoints are not used by the Android app.
 */
object Prefs {
    const val NAME = "orion"
    const val HOST = "host"
    const val PORT = "port"
    const val HTTPS = "https"
    const val ENDPOINTS = "endpoints"     // JSON array of base URLs, priority order
    const val DEVICE = "device_id"
    const val REFRESH = "refresh_token"           // legacy plain text, migrated on read
    const val REFRESH_SEALED = "refresh_token_sealed"   // Keystore AES-GCM ciphertext
    const val LITE = "lite"               // "auto" (metered-driven) | "on" | "off"
    const val CERT_PIN = "cert_sha256"    // SHA-256 of ORION's self-signed certificate
    const val DEFAULT_PORT = 443

    private fun sp(context: Context) =
        context.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    /** The manually-configured base URL, or null if none was ever entered. */
    fun baseUrl(context: Context): String? {
        val p = sp(context)
        val host = p.getString(HOST, "").orEmpty().trim()
        if (host.isEmpty()) return null
        val savedPort = p.getInt(PORT, DEFAULT_PORT)
        val port = if (host.endsWith(".ts.net", ignoreCase = true) &&
            savedPort in setOf(8765, 8000)) 443 else savedPort
        val scheme = "https"
        // An IPv6 literal (Tailscale hands out fd7a:…) must be bracketed in a URL.
        val hostPart = if (host.contains(':') && !host.startsWith("[")) "[$host]" else host
        return secureUrl("$scheme://$hostPart:$port/")
    }

    /** What the user typed into "PC address", split into its parts. */
    data class Address(val host: String, val port: Int?, val https: Boolean?)

    /**
     * Accept what people actually paste: `192.168.1.5`, `192.168.1.5:8765`,
     * `https://orion-pc.tail1234.ts.net:8765/`. Storing the raw text as the host
     * turned a pasted URL into `https://https://…:8765:8765/`, which can never
     * connect. Returns null when there is no usable host or the port is invalid.
     */
    fun parseAddress(raw: String): Address? {
        var s = raw.trim()
        if (s.isEmpty()) return null
        var https: Boolean? = null
        val schemeEnd = s.indexOf("://")
        if (schemeEnd >= 0) {
            https = when (s.substring(0, schemeEnd).lowercase()) {
                "https" -> true
                "http" -> false
                else -> return null
            }
            s = s.substring(schemeEnd + 3)
        }
        if ('@' in s || '?' in s || '#' in s) return null
        val authority = s.substringBefore('/')
        if (s.substring(authority.length) !in setOf("", "/")) return null
        s = authority
        var port: Int? = null
        val host: String
        if (s.startsWith("[")) {                       // [IPv6]:port
            val close = s.indexOf(']')
            if (close < 0) return null
            host = s.substring(1, close)
            val rest = s.substring(close + 1)
            if (rest.startsWith(":")) port = rest.substring(1).toIntOrNull() ?: return null
            else if (rest.isNotEmpty()) return null
        } else if (s.count { it == ':' } == 1) {       // host:port
            host = s.substringBefore(':')
            port = s.substringAfter(':').toIntOrNull() ?: return null
        } else {                                       // bare host or bare IPv6
            host = s
        }
        if (host.isEmpty() || host.any { it.isWhitespace() }) return null
        if (port != null && port !in 1..65535) return null
        val hostPart = if (host.contains(':')) "[$host]" else host
        if (canonicalHttpsOrigin("https://$hostPart:${port ?: DEFAULT_PORT}/") == null) return null
        return Address(host, port, https)
    }

    fun host(context: Context): String =
        sp(context).getString(HOST, "").orEmpty().trim()

    /** Ordered HTTPS base URLs: learned endpoints first, then the manual host. */
    fun candidateUrls(context: Context): List<String> {
        val out = ArrayList<String>()
        val raw = sp(context).getString(ENDPOINTS, "").orEmpty()
        if (raw.isNotEmpty()) {
            try {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) {
                    val url = arr.optString(i).trim()
                secureUrl(url)?.let { if (!out.contains(it)) out.add(it) }
                }
            } catch (_: Exception) {
            }
        }
        baseUrl(context)?.let { if (!out.contains(it)) out.add(it) }
        return out
    }

    /** Hosts whose self-signed certificate the WebView should trust — every
     *  host we've been told about or the user configured. */
    fun trustedHosts(context: Context): Set<String> {
        val hosts = HashSet<String>()
        host(context).takeIf { it.isNotEmpty() }?.let { hosts.add(it) }
        for (u in candidateUrls(context)) {
            val h = hostOf(u)
            if (h.isNotEmpty()) hosts.add(h)
        }
        return hosts
    }

    fun hostOf(url: String): String = try {
        Uri.parse(url).host.orEmpty()
    } catch (_: Exception) {
        ""
    }

    internal fun canonicalHttpsOrigin(value: String): String? {
        if (value.length > 2048 || value.any { it.isWhitespace() || it.code < 32 }) return null
        val parsed = value.toHttpUrlOrNull() ?: return null
        if (parsed.scheme != "https" || parsed.username.isNotEmpty() || parsed.password.isNotEmpty() ||
            parsed.encodedPath != "/" || parsed.query != null || parsed.fragment != null ||
            parsed.host.matches(Regex("100\\.\\d+\\.\\d+\\.\\d+"))) return null
        val port = if (parsed.host.endsWith(".ts.net", true) && parsed.port in setOf(8765, 8000))
            DEFAULT_PORT else parsed.port
        return parsed.newBuilder().port(port).build().toString()
    }

    private fun secureUrl(url: String): String? = canonicalHttpsOrigin(url)

    /** Persist the endpoint list the desktop reported (from /v1/auth/pair or
     *  /v1/auth/token). Accepts a JSON array of {url,…} objects or bare strings.
     *
     *  Only ever called with an answer that arrived over a trusted HTTPS
     *  connection (system CA or an existing pin) and authenticated by the
     *  pairing: that is what makes a home Wi-Fi endpoint's `cert_sha256` safe
     *  to pin. A pin is accepted only for a private LAN IPv4 literal. */
    fun saveEndpoints(context: Context, endpointsJson: String) {
        val urls = JSONArray()
        try {
            val arr = JSONArray(endpointsJson)
            for (i in 0 until arr.length()) {
                val item = arr.opt(i)
                val url = when (item) {
                    is JSONObject -> item.optString("url").trim()
                    else -> item.toString().trim()
                }
                val secure = secureUrl(url) ?: continue
                urls.put(secure)
                if (item is JSONObject && item.has("cert_sha256")) {
                    val parsed = secure.toHttpUrlOrNull() ?: continue
                    if (LanTrust.isPrivateLanHost(parsed.host))
                        LanTrust.savePin(context, parsed.host, parsed.port, item.optString("cert_sha256"))
                }
            }
        } catch (_: Exception) {
            return
        }
        if (urls.length() > 0) {
            sp(context).edit().putString(ENDPOINTS, urls.toString()).apply()
        }
    }

    /** Put a home Wi-Fi address found by [LanDiscovery] (its certificate
     *  already matched a pin) at the front of the learned endpoints. */
    fun preferEndpoint(context: Context, url: String) {
        val secure = secureUrl(url) ?: return
        val next = JSONArray().put(secure)
        try {
            val arr = JSONArray(sp(context).getString(ENDPOINTS, "[]").orEmpty())
            for (i in 0 until arr.length()) {
                val old = arr.optString(i)
                val host = old.toHttpUrlOrNull()?.host.orEmpty()
                // A LAN address that moved is replaced, not kept beside the new one.
                if (old != secure && !LanTrust.isPrivateLanHost(host)) next.put(old)
            }
        } catch (_: Exception) { }
        sp(context).edit().putString(ENDPOINTS, next.toString()).apply()
    }

    /** Manual reconfiguration clears learned endpoints so the new host wins. */
    fun clearEndpoints(context: Context) {
        sp(context).edit().remove(ENDPOINTS).apply()
    }

    /**
     * The pairing (device id + refresh token) the uplink page issued. The page
     * keeps it in localStorage, which is per ORIGIN — so a phone paired over the
     * LAN address was unpaired on the Tailscale address. Holding a copy here lets
     * MainActivity hand it to every ORION origin, making "pair once" true.
     */
    fun credentials(context: Context): Pair<String, String>? {
        val p = sp(context)
        val device = p.getString(DEVICE, "").orEmpty()
        if (device.isEmpty()) return null
        val sealed = p.getString(REFRESH_SEALED, "").orEmpty()
        if (sealed.isNotEmpty()) {
            cached?.let { (text, plain) -> if (text == sealed) return device to plain }
            val refresh = SecureStore.decrypt(sealed)
            if (refresh.isNullOrEmpty()) {
                // The key is gone or the value was altered. An unreadable
                // credential must never look like a valid session: forget it
                // and let the app pair again, as on a fresh install.
                clearCredentials(context)
                return null
            }
            cached = sealed to refresh
            return device to refresh
        }
        // Stored in plain text by an earlier version: seal it now.
        val legacy = p.getString(REFRESH, "").orEmpty()
        if (legacy.isEmpty()) return null
        saveCredentials(context, device, legacy)
        return if (sp(context).getString(REFRESH_SEALED, "").isNullOrEmpty()) null else device to legacy
    }

    /** The refresh credential is stored only as Keystore ciphertext. If the
     *  Keystore refuses, nothing is stored (pair again) rather than falling
     *  back to plain text. */
    fun saveCredentials(context: Context, deviceId: String, refreshToken: String) {
        val sealed = SecureStore.encrypt(refreshToken)
        val edit = sp(context).edit().putString(DEVICE, deviceId).remove(REFRESH)
        if (sealed != null) {
            edit.putString(REFRESH_SEALED, sealed)
            cached = sealed to refreshToken
        } else {
            edit.remove(REFRESH_SEALED)
            cached = null
        }
        edit.apply()
    }

    fun clearCredentials(context: Context) {
        cached = null
        sp(context).edit().remove(DEVICE).remove(REFRESH).remove(REFRESH_SEALED).apply()
    }

    /** (ciphertext, plain) of the last credential read or written, so the
     *  frame uplink and page loads do not pay a Keystore round trip each time. */
    @Volatile
    private var cached: Pair<String, String>? = null

    /**
     * The certificate ORION's uplink presented the first time this phone
     * accepted it. His certificate is self-signed and lasts for years, so
     * trusting "whatever a known host name presents" let anyone on the same
     * network impersonate him and collect the pairing token. After the first
     * acceptance only this exact certificate is trusted. Empty until then.
     */
    fun certPin(context: Context): String =
        sp(context).getString(CERT_PIN, "").orEmpty()

    fun setCertPin(context: Context, sha256: String) {
        sp(context).edit().putString(CERT_PIN, sha256).apply()
    }

    /** Forget the pin — a deliberate reconfiguration, e.g. after ORION's
     *  certificate was regenerated. */
    fun clearCertPin(context: Context) {
        sp(context).edit().remove(CERT_PIN).apply()
    }

    fun litePreference(context: Context): String =
        sp(context).getString(LITE, "auto").orEmpty()

    fun setLitePreference(context: Context, value: String) {
        sp(context).edit().putString(LITE, value).apply()
    }
}
