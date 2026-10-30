package com.orion.companion

import android.Manifest
import android.app.AlarmManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat

/**
 * The nudges of a focus block - the break when it is up, the check-in halfway
 * through when ORION is body doubling, the end of a break. Alarms, so they
 * arrive with the app closed and the screen off; exact when Android allows
 * it, otherwise within Android's own window.
 */
internal object FocusNudges {
    private const val CHANNEL = "orion_focus"
    private const val ACTION = "com.orion.companion.FOCUS_NUDGE"
    /** Request codes: one slot per kind of nudge, so a new block replaces the old. */
    private val SLOTS = mapOf("break" to 4101, "halfway" to 4102, "back" to 4103)

    fun schedule(context: Context, kind: String, text: String, atMs: Long) {
        val slot = SLOTS[kind] ?: return
        val alarms = context.getSystemService(AlarmManager::class.java) ?: return
        val pending = intent(context, slot, text)
        val exact = Build.VERSION.SDK_INT < Build.VERSION_CODES.S || alarms.canScheduleExactAlarms()
        runCatching {
            if (exact) alarms.setExactAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, atMs, pending)
            else alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, atMs, pending)
        }.onFailure { alarms.setAndAllowWhileIdle(AlarmManager.RTC_WAKEUP, atMs, pending) }
    }

    fun cancelAll(context: Context) {
        val alarms = context.getSystemService(AlarmManager::class.java) ?: return
        SLOTS.values.forEach { alarms.cancel(intent(context, it, "")) }
    }

    private fun intent(context: Context, slot: Int, text: String): PendingIntent =
        PendingIntent.getBroadcast(context, slot,
            Intent(context, FocusNudgeReceiver::class.java).setAction(ACTION).putExtra("text", text),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)

    fun notify(context: Context, text: String) {
        if (text.isBlank()) return
        if (Build.VERSION.SDK_INT >= 33 && context.checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED) return
        val manager = context.getSystemService(NotificationManager::class.java) ?: return
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "Focus",
            NotificationManager.IMPORTANCE_HIGH).apply { description = "Breaks and check-ins during a focus block" })
        val open = PendingIntent.getActivity(context, 4100,
            Intent(context, DayActivity::class.java).putExtra(DayActivity.EXTRA_TAB, DayActivity.FOCUS)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        manager.notify(4100, NotificationCompat.Builder(context, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_orion)
            .setContentTitle("ORION")
            .setContentText(text)
            .setCategory(NotificationCompat.CATEGORY_REMINDER)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setContentIntent(open)
            .setAutoCancel(true)
            .build())
    }
}

class FocusNudgeReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        FocusNudges.notify(context, intent.getStringExtra("text").orEmpty())
    }
}
