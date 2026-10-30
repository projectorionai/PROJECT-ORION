package com.orion.companion

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import androidx.core.content.ContextCompat

/** Human-readable scope of native actions and their actual completion state. */
internal object PhoneCapabilities {
    fun summary(context: Context): String {
        val voice = if (PhoneAiProfile.hasKey(context)) "set up" else "needs your Gemini key"
        val text = if (PhoneHostedProfile.ready(context)) "set up" else "needs a text AI key"
        val contacts = if (ContextCompat.checkSelfPermission(context, Manifest.permission.READ_CONTACTS) ==
            PackageManager.PERMISSION_GRANTED) "available" else "needs Contacts permission"
        val sms = if (ContextCompat.checkSelfPermission(context, Manifest.permission.SEND_SMS) ==
            PackageManager.PERMISSION_GRANTED) "Android accepts submission; delivery is unverified"
            else "opens a Messages draft for you to send"
        val gmail = if (context.packageManager.getLaunchIntentForPackage(PhoneActionRules.GMAIL_PACKAGE) == null)
            "reviewed sending needs Gmail installed"
            else if (OrionAccessibilityService.isConnected()) "reviewed Gmail sending is available"
            else "reviewed Gmail sending needs Phone accessibility"
        val maps = if (listOf("net.osmand", "net.osmand.plus").any {
                context.packageManager.getLaunchIntentForPackage(it) != null
            }) "OsmAnd installed; download the map and start the route there"
            else "install OsmAnd and download maps before driving"
        val mcp = if (PhoneMcpClient.configured(context)) "one HTTPS service configured"
            else "no HTTPS service configured"
        return "REPLIES\nText AI: $text. Microphone and Charon: $voice.\n\n" +
            "PHONE\nOpen installed apps and links; read battery and connection status after review. " +
            "Look up contacts: $contacts. Calls can be started, but connection is unverified. " +
            "SMS: $sms. E-mail: $gmail; other mail clients open a draft.\n\n" +
            "HANDOFFS\nAlarms, timers, contacts and calendar events open drafts or their apps for your final check. " +
            "Offline UK map: native 2D road view and live position once map packs and Location permission are available. " +
            "Native car route calculation and a route overlay need BRouter, its car-fast profile, every journey's " +
            "downloaded routing segment and a fresh location. GPX route import and saved coordinate destinations " +
            "are available in the map. Offline address search and spoken or locked-screen turn guidance: $maps. " +
            "A map-screen launch does not confirm a route was calculated or navigation started.\n\n" +
            "EXTENSIONS\nPhone MCP: $mcp. Options → Connections → PC controls can shut down your paired PC " +
            "after confirmation, with a 60-second cancellation window. The PC must stay online with ORION running. " +
            "Other host controls are available in Desktop mode."
    }
}
