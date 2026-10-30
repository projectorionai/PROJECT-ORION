package com.orion.companion

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.SystemClock
import java.net.Inet4Address

/** What the phone's connection can do right now, cheaply (read at most once a second). */
internal object PhoneNetwork {
    @Volatile private var checkedAt = 0L
    @Volatile private var lastOnline = true

    /** True when Android has validated an internet path (Wi-Fi or mobile). */
    fun online(context: Context): Boolean {
        val now = SystemClock.elapsedRealtime()
        if (now - checkedAt < 1_000L) return lastOnline
        lastOnline = try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val caps = cm.activeNetwork?.let(cm::getNetworkCapabilities)
            caps != null && caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) &&
                caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        } catch (_: Exception) { true }
        checkedAt = now
        return lastOnline
    }

    /** The phone's own IPv4 address and prefix on Wi-Fi or Ethernet - the
     *  network the PC can be on - or null on mobile data / no network. */
    fun localSubnet(context: Context): Pair<Inet4Address, Int>? = try {
        val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        cm.allNetworks.firstNotNullOfOrNull { network ->
            val caps = cm.getNetworkCapabilities(network) ?: return@firstNotNullOfOrNull null
            if (!caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
                !caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET)) return@firstNotNullOfOrNull null
            cm.getLinkProperties(network)?.linkAddresses?.firstNotNullOfOrNull { link ->
                (link.address as? Inet4Address)?.takeIf { LanTrust.isPrivateLanHost(it.hostAddress.orEmpty()) }
                    ?.let { it to link.prefixLength }
            }
        }
    } catch (_: Exception) { null }
}
