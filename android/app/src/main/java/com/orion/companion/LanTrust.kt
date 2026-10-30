package com.orion.companion

import android.content.Context
import android.net.Uri
import android.net.http.SslCertificate
import android.os.Build
import okhttp3.OkHttpClient
import okhttp3.internal.tls.OkHostnameVerifier
import org.json.JSONObject
import java.io.IOException
import java.net.InetSocketAddress
import java.net.Socket
import java.security.KeyStore
import java.security.MessageDigest
import java.security.cert.CertificateException
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLEngine
import javax.net.ssl.SSLSession
import javax.net.ssl.SSLSocket
import javax.net.ssl.TrustManagerFactory
import javax.net.ssl.X509ExtendedTrustManager
import javax.net.ssl.X509TrustManager

/**
 * Trust for the PC's home Wi-Fi link (orion_core/remote_lan.py on the PC).
 *
 * On the LAN ORION serves HTTPS with his own self-signed certificate, which no
 * public CA vouches for. The phone therefore PINS it: it trusts a private-LAN
 * address only when the server presents the exact certificate whose SHA-256
 * it was given
 *   - inside an authenticated /v1/auth/pair or /v1/auth/token answer that
 *     arrived over an already trusted connection (Tailscale's public
 *     certificate, a tunnel, or an existing pin), or
 *   - by the owner typing the certificate code the PC shows beside the
 *     pairing code (Security Centre), checked against the certificate the PC
 *     actually presents before anything is sent.
 * Every named host (Tailscale, a tunnel) keeps Android's normal CA and host
 * checks exactly as before; a pin never vouches for a public name. A LAN
 * address with no pin is refused like any untrusted certificate.
 */
internal object LanTrust {
    private const val PINS = "lan_pins"            // JSON {"host:port": sha256}
    private val SHA256 = Regex("[0-9a-f]{64}")

    // ── pure rules (unit-tested) ─────────────────────────────────────────────

    /** An IPv4 literal on a home network: 10/8, 172.16/12 or 192.168/16.
     *  Never a name (no DNS answer can make a host "local"), never
     *  Tailscale's 100.64/10, never loopback. */
    fun isPrivateLanHost(host: String): Boolean {
        val parts = host.split('.')
        if (parts.size != 4) return false
        val octets = parts.map { part ->
            if (part.isEmpty() || part.length > 3 || !part.all(Char::isDigit)) return false
            part.toInt().takeIf { it in 0..255 } ?: return false
        }
        return octets[0] == 10 || (octets[0] == 172 && octets[1] in 16..31) ||
            (octets[0] == 192 && octets[1] == 168)
    }

    fun fingerprint(der: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(der).joinToString("") { "%02x".format(it) }

    /** The code the PC shows: the first 64 bits in four groups (1A2B-3C4D-...). */
    fun displayCode(sha256: String): String =
        sha256.uppercase().filter { it in '0'..'9' || it in 'A'..'F' }.take(16).chunked(4).joinToString("-")

    /** The owner typed at least the 16-character code, and it is the start of
     *  this certificate's fingerprint. Dashes, spaces and case do not matter. */
    fun codeMatches(sha256: String, entered: String): Boolean {
        val typed = entered.uppercase().filter { it in '0'..'9' || it in 'A'..'F' }
        return typed.length >= 16 && SHA256.matches(sha256) && sha256.uppercase().startsWith(typed)
    }

    fun key(host: String, port: Int): String = "${host.lowercase()}:$port"

    /** Pins from their stored JSON; anything malformed is simply not a pin. */
    fun parsePins(json: String): Map<String, String> = try {
        val obj = JSONObject(json)
        obj.keys().asSequence().mapNotNull { k ->
            val host = k.substringBeforeLast(':')
            val port = k.substringAfterLast(':').toIntOrNull()
            val sha = obj.optString(k).lowercase()
            if (port != null && port in 1..65535 && isPrivateLanHost(host) && SHA256.matches(sha))
                key(host, port) to sha else null
        }.toMap()
    } catch (_: Exception) { emptyMap() }

    fun withPin(json: String, host: String, port: Int, sha256: String): String? {
        val sha = sha256.lowercase()
        if (!isPrivateLanHost(host) || port !in 1..65535 || !SHA256.matches(sha)) return null
        val pins = parsePins(json).toMutableMap()
        pins[key(host, port)] = sha
        return JSONObject(pins as Map<*, *>).toString()
    }

    // ── stored pins ──────────────────────────────────────────────────────────

    private fun prefs(context: Context) =
        context.applicationContext.getSharedPreferences(Prefs.NAME, Context.MODE_PRIVATE)

    fun pins(context: Context): Map<String, String> =
        parsePins(prefs(context).getString(PINS, "{}").orEmpty())

    fun pinFor(context: Context, host: String, port: Int): String? = pins(context)[key(host, port)]

    /** Remember a pin. Only for a private LAN address and a well-formed hash. */
    fun savePin(context: Context, host: String, port: Int, sha256: String): Boolean {
        val next = withPin(prefs(context).getString(PINS, "{}").orEmpty(), host, port, sha256) ?: return false
        return prefs(context).edit().putString(PINS, next).commit()
    }

    fun clearPins(context: Context) { prefs(context).edit().remove(PINS).apply() }

    // ── TLS ──────────────────────────────────────────────────────────────────

    /** OkHttp with Android's normal checks, plus exact-pin trust for pinned
     *  LAN addresses only. */
    fun secure(builder: OkHttpClient.Builder, context: Context): OkHttpClient.Builder {
        val app = context.applicationContext
        val factory = TrustManagerFactory.getInstance(TrustManagerFactory.getDefaultAlgorithm())
        factory.init(null as KeyStore?)
        val system = factory.trustManagers.filterIsInstance<X509TrustManager>().first()
        val trust = PinningTrustManager(app, system)
        val tls = SSLContext.getInstance("TLS").apply { init(null, arrayOf(trust), null) }
        return builder.sslSocketFactory(tls.socketFactory, trust)
            .hostnameVerifier(PinningHostnameVerifier(app, OkHostnameVerifier))
    }

    private fun pinnedMatch(app: Context, chain: Array<out X509Certificate>?, host: String?, port: Int): Boolean {
        if (host == null || !isPrivateLanHost(host)) return false
        val pin = pinFor(app, host, port) ?: return false
        val leaf = chain?.firstOrNull() ?: throw CertificateException("ORION's LAN link sent no certificate")
        if (fingerprint(leaf.encoded) != pin)
            throw CertificateException("This PC's certificate is not the one paired with ORION")
        return true
    }

    private class PinningTrustManager(private val app: Context, private val system: X509TrustManager) :
        X509ExtendedTrustManager() {
        override fun getAcceptedIssuers(): Array<X509Certificate> = system.acceptedIssuers
        override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) =
            system.checkClientTrusted(chain, authType)
        override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?, socket: Socket?) =
            system.checkClientTrusted(chain, authType)
        override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?, engine: SSLEngine?) =
            system.checkClientTrusted(chain, authType)
        // No host known here: only Android's ordinary chain validation applies.
        override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) =
            system.checkServerTrusted(chain, authType)

        override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?, socket: Socket?) {
            val host = (socket as? SSLSocket)?.handshakeSession?.peerHost ?: socket?.inetAddress?.hostAddress
            if (pinnedMatch(app, chain, host, socket?.port ?: -1)) return
            if (system is X509ExtendedTrustManager) system.checkServerTrusted(chain, authType, socket)
            else system.checkServerTrusted(chain, authType)
        }

        override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?, engine: SSLEngine?) {
            if (pinnedMatch(app, chain, engine?.peerHost, engine?.peerPort ?: -1)) return
            if (system is X509ExtendedTrustManager) system.checkServerTrusted(chain, authType, engine)
            else system.checkServerTrusted(chain, authType)
        }
    }

    /** A pinned LAN address is verified by its certificate (its name list was
     *  written for the PC's address on the day the certificate was made);
     *  every other host by the normal host-name rules. */
    private class PinningHostnameVerifier(private val app: Context, private val system: HostnameVerifier) :
        HostnameVerifier {
        override fun verify(host: String, session: SSLSession): Boolean {
            if (isPrivateLanHost(host)) {
                val pin = pinFor(app, host, session.peerPort)
                if (pin != null) {
                    val leaf = runCatching { session.peerCertificates.firstOrNull() as? X509Certificate }.getOrNull()
                    return leaf != null && fingerprint(leaf.encoded) == pin
                }
            }
            return system.verify(host, session)
        }
    }

    /** The WebView (Desktop mode) on a pinned LAN address: proceed only for
     *  the exact pinned certificate. */
    fun webViewCertificateMatches(context: Context, url: String?, certificate: SslCertificate?): Boolean {
        val uri = runCatching { Uri.parse(url.orEmpty()) }.getOrNull() ?: return false
        val host = uri.host.orEmpty()
        if (uri.scheme != "https" || !isPrivateLanHost(host) || certificate == null) return false
        val pin = pinFor(context, host, if (uri.port > 0) uri.port else 443) ?: return false
        val der = try {
            if (Build.VERSION.SDK_INT >= 29) certificate.x509Certificate?.encoded
            else SslCertificate.saveState(certificate).getByteArray("x509-certificate")
        } catch (_: Exception) { null } ?: return false
        return fingerprint(der) == pin
    }

    /**
     * The SHA-256 of the certificate a LAN address presents, read during the
     * TLS handshake. Nothing is sent: the capture refuses the certificate the
     * moment it has seen it, so the handshake never completes and no request,
     * token or code can follow on this socket.
     */
    fun presentedFingerprint(host: String, port: Int, timeoutMs: Int = 2_500): String {
        if (!isPrivateLanHost(host)) throw IOException("Only a home-network address is checked this way.")
        var seen: X509Certificate? = null
        val capture = object : X509TrustManager {
            override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
            override fun checkClientTrusted(chain: Array<out X509Certificate>?, authType: String?) =
                throw CertificateException("no client certificates")
            override fun checkServerTrusted(chain: Array<out X509Certificate>?, authType: String?) {
                seen = chain?.firstOrNull()
                throw CertificateException("certificate read; not trusted by this check")
            }
        }
        val tls = SSLContext.getInstance("TLS").apply { init(null, arrayOf(capture), null) }
        Socket().use { raw ->
            raw.connect(InetSocketAddress(host, port), timeoutMs)
            raw.soTimeout = timeoutMs
            (tls.socketFactory.createSocket(raw, host, port, true) as SSLSocket).use { socket ->
                runCatching { socket.startHandshake() }
            }
        }
        val leaf = seen ?: throw IOException("The PC did not present a certificate.")
        return fingerprint(leaf.encoded)
    }

    /** Parse a DER certificate (tests and the pre-Q WebView path). */
    internal fun x509(der: ByteArray): X509Certificate =
        CertificateFactory.getInstance("X.509").generateCertificate(der.inputStream()) as X509Certificate
}
