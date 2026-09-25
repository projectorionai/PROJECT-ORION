package com.orion.companion

import android.content.Context
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject

/**
 * Where the phone remembers how to reach ORION.
 *
 * Beyond the single manually-typed host, it stores the ordered **endpoint list**
 * the desktop reports at pairing time (Tailscale name first, then LAN). The app
 * auto-connects through these, so once you've paired you never type an IP again:
 * the Tailscale name reaches ORION at home and on mobile data alike, and the LAN
 * address is a low-latency fallback when you're on his Wi-Fi.
 */
object Prefs {
    const val NAME = "orion"
    const val HOST = "host"
    const val PORT = "port"
    const val HTTPS = "https"
    const val ENDPOINTS = "endpoints"     // JSON array of base URLs, priority order
    const val DEVICE = "device_id"
    const val REFRESH = "refresh_token"
    const val LITE = "lite"               // "auto" (metered-driven) | "on" | "off"
    const val CERT_PIN = "cert_sha256"    // SHA-256 of ORION's self-signed certificate
    const val DEFAULT_PORT = 8765

    private fun sp(context: Context) =
        context.getSharedPreferences(NAME, Context.MODE_PRIVATE)

    /** The manually-configured base URL, or null if none was ever entered. */
    fun baseUrl(context: Context): String? {
        val p = sp(context)
        val host = p.getString(HOST, "").orEmpty().trim()
        if (host.isEmpty()) return null
        val port = p.getInt(PORT, DEFAULT_PORT)
        val scheme = if (p.getBoolean(HTTPS, true)) "https" else "http"
        // An IPv6 literal (Tailscale hands out fd7a:…) must be bracketed in a URL.
        val hostPart = if (host.contains(':') && !host.startsWith("[")) "[$host]" else host
        return "$scheme://$hostPart:$port/"
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
        s = s.substringBefore('/').substringBefore('?').substringBefore('#').substringAfterLast('@')
        var port: Int? = null
        val host: String
        if (s.startsWith("[")) {                       // [IPv6]:port
            val close = s.indexOf(']')
            if (close < 0) return null
            host = s.substring(1, close)
            val rest = s.substring(close + 1)
            if (rest.startsWith(":")) port = rest.substring(1).toIntOrNull() ?: return null
        } else if (s.count { it == ':' } == 1) {       // host:port
            host = s.substringBefore(':')
            port = s.substringAfter(':').toIntOrNull() ?: return null
        } else {                                       // bare host or bare IPv6
            host = s
        }
        if (host.isEmpty() || host.any { it.isWhitespace() }) return null
        if (port != null && port !in 1..65535) return null
        return Address(host, port, https)
    }

    fun host(context: Context): String =
        sp(context).getString(HOST, "").orEmpty().trim()

    /** Ordered base URLs to try: learned endpoints first, then the manual host. */
    fun candidateUrls(context: Context): List<String> {
        val out = ArrayList<String>()
        val raw = sp(context).getString(ENDPOINTS, "").orEmpty()
        if (raw.isNotEmpty()) {
            try {
                val arr = JSONArray(raw)
                for (i in 0 until arr.length()) {
                    val url = arr.optString(i).trim()
                    if (url.isNotEmpty() && !out.contains(url)) out.add(url)
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

    /** Persist the endpoint list the desktop reported (from /v1/auth/pair or
     *  /v1/auth/token). Accepts a JSON array of {url,…} objects or bare strings. */
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
                if (url.isNotEmpty()) urls.put(url)
            }
        } catch (_: Exception) {
            return
        }
        if (urls.length() > 0) {
            sp(context).edit().putString(ENDPOINTS, urls.toString()).apply()
        }
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
        val refresh = p.getString(REFRESH, "").orEmpty()
        return if (device.isNotEmpty() && refresh.isNotEmpty()) device to refresh else null
    }

    fun saveCredentials(context: Context, deviceId: String, refreshToken: String) {
        sp(context).edit().putString(DEVICE, deviceId).putString(REFRESH, refreshToken).apply()
    }

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
