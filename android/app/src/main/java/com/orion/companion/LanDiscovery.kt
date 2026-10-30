package com.orion.companion

import android.content.Context
import android.os.SystemClock
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Finds the PC again on the home Wi-Fi when its LAN address changes (a DHCP
 * renewal, a new router) - without Tailscale and without typing an address.
 *
 * It only looks for a PC this phone has ALREADY paired with: each host on the
 * phone's own Wi-Fi subnet that answers on a pinned port is asked for its
 * certificate (LanTrust.presentedFingerprint - nothing is sent), and only a
 * host presenting exactly the pinned certificate is adopted. A stranger on the
 * network cannot pose as ORION this way; it would need ORION's private key.
 */
internal object LanDiscovery {
    private const val MIN_INTERVAL_MS = 120_000L
    private const val CONNECT_MS = 300
    @Volatile private var lastScanAt = 0L

    /** Every other host address of [own]'s subnet, nearest first. Subnets
     *  wider than /22 are searched as the phone's own /24 only. */
    fun subnetHosts(own: ByteArray, prefix: Int): List<String> {
        require(own.size == 4)
        val ip = own.fold(0L) { acc, b -> (acc shl 8) or (b.toLong() and 0xff) }
        val bits = if (prefix in 22..30) prefix else 24
        val mask = (0xffffffffL shl (32 - bits)) and 0xffffffffL
        val network = ip and mask
        val broadcast = network or (mask.inv() and 0xffffffffL)
        return ((network + 1) until broadcast).filter { it != ip }
            .sortedBy { kotlin.math.abs(it - ip) }
            .map { "${(it shr 24) and 0xff}.${(it shr 16) and 0xff}.${(it shr 8) and 0xff}.${it and 0xff}" }
    }

    /**
     * Blocking; call off the main thread. Returns the adopted base URL, or
     * null when nothing was found, the phone is not on Wi-Fi, no LAN pin is
     * known, or a scan ran less than two minutes ago.
     */
    fun rediscover(context: Context): String? {
        val app = context.applicationContext
        val pins = LanTrust.pins(app)
        if (pins.isEmpty()) return null
        val now = SystemClock.elapsedRealtime()
        if (now - lastScanAt < MIN_INTERVAL_MS) return null
        lastScanAt = now
        val (own, prefix) = PhoneNetwork.localSubnet(app) ?: return null
        val wanted = pins.values.toSet()
        val ports = pins.keys.mapNotNull { it.substringAfterLast(':').toIntOrNull() }.toSet()
        val found = AtomicReference<Pair<String, Int>?>(null)
        val pool = Executors.newFixedThreadPool(32) { job -> Thread(job, "orion-lan-scan").apply { isDaemon = true } }
        try {
            for (host in subnetHosts(own.address, prefix)) for (port in ports) pool.execute {
                if (found.get() != null) return@execute
                val open = try {
                    Socket().use { it.connect(InetSocketAddress(host, port), CONNECT_MS); true }
                } catch (_: Exception) { false }
                if (!open || found.get() != null) return@execute
                val sha = runCatching { LanTrust.presentedFingerprint(host, port) }.getOrNull()
                if (sha != null && sha in wanted) found.compareAndSet(null, host to port)
            }
            pool.shutdown()
            pool.awaitTermination(20, TimeUnit.SECONDS)
        } finally {
            pool.shutdownNow()
        }
        val (host, port) = found.get() ?: return null
        val sha = runCatching { LanTrust.presentedFingerprint(host, port) }.getOrNull()
        if (sha == null || sha !in wanted) return null       // checked twice, never assumed
        LanTrust.savePin(app, host, port, sha)
        val url = "https://$host:$port/"
        Prefs.preferEndpoint(app, url)
        return url
    }
}
