package com.orion.companion

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import androidx.core.app.NotificationCompat
import org.json.JSONObject
import java.util.UUID

/** Bring a background phone task to the same native approval and receipt path. */
object PhoneActionCoordinator {
    data class Request(val id: String, val action: JSONObject, val expiresAtMs: Long,
        val report: (JSONObject) -> Unit)
    private val main = Handler(Looper.getMainLooper())
    private var pending: Request? = null
    private var handler: ((Request) -> Unit)? = null
    private const val CHANNEL = "orion_phone_actions"
    // A distinct ID is vital: 407 belongs to the microphone foreground
    // service. Replacing or cancelling it can stop hands-free capture.
    private const val NOTICE = 408
    private const val REVIEW_TIMEOUT_MS = 120_000L
    fun request(context: Context, action: JSONObject, report: (JSONObject) -> Unit) {
        val copy = JSONObject(action.toString())
        main.post {
            val brainRequest = copy.optString("_phone_brain_request")
            if (brainRequest.isNotBlank() && (!PhoneBrain.isRequestActive(brainRequest) ||
                    SystemClock.elapsedRealtime() >= copy.optLong("_phone_brain_deadline", 0))) {
                report(JSONObject().put("ok", false).put("detail", "The phone request expired. Nothing has run."))
                return@post
            }
            if (pending != null) { report(JSONObject().put("ok", false).put("detail", "Another phone action is awaiting review.")); return@post }
            val task = Request(UUID.randomUUID().toString(), copy,
                SystemClock.elapsedRealtime() + REVIEW_TIMEOUT_MS, report)
            pending = task
            val active = handler
            if (active != null) active(task) else if (!notify(context)) {
                pending = null
                task.report(JSONObject().put("ok", false)
                    .put("detail", "Enable ORION notifications, then try the phone action again. Android could not show its review request."))
                return@post
            }
            main.postDelayed({ if (pending?.id == task.id) complete(context, task.id,
                JSONObject().put("ok", false).put("detail", "Phone action review expired. Nothing further has run.")) }, REVIEW_TIMEOUT_MS)
        }
    }
    fun attach(context: Context, actionHandler: (Request) -> Unit) {
        main.post { handler = actionHandler; pending?.let(actionHandler)
            (context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTICE) }
    }
    fun detach(actionHandler: (Request) -> Unit) { main.post { if (handler === actionHandler) handler = null } }
    fun current(id: String): Boolean = pending?.let {
        it.id == id && SystemClock.elapsedRealtime() < it.expiresAtMs
    } ?: false
    fun complete(context: Context, id: String, receipt: JSONObject) {
        main.post {
            val task = pending?.takeIf { it.id == id } ?: return@post
            pending = null
            (context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTICE)
            task.report(receipt)
        }
    }
    private fun notify(context: Context): Boolean {
        val manager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.createNotificationChannel(NotificationChannel(CHANNEL, "ORION phone tasks", NotificationManager.IMPORTANCE_DEFAULT))
        if (!manager.areNotificationsEnabled() ||
            manager.getNotificationChannel(CHANNEL)?.importance == NotificationManager.IMPORTANCE_NONE) return false
        val open = PendingIntent.getActivity(context, NOTICE, Intent(context, PhoneConsoleActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        val notice = NotificationCompat.Builder(context, CHANNEL).setSmallIcon(R.drawable.ic_stat_orion)
            .setContentTitle("ORION has a phone action to review").setContentText("Open ORION to review the exact action. It has not run.")
            .setContentIntent(open).setAutoCancel(true).build()
        return runCatching { manager.notify(NOTICE, notice) }.isSuccess
    }
    fun cancelAll(context: Context) {
        main.post {
            val task = pending ?: return@post
            pending = null
            (context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).cancel(NOTICE)
            task.report(JSONObject().put("ok", false)
                .put("detail", "The action was cancelled. Later steps have not run."))
        }
    }
}
