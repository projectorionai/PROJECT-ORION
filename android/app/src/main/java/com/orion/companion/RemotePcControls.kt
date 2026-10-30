package com.orion.companion

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONObject
import java.io.Closeable
import java.util.concurrent.Executors

/** Optional PC controls alongside the phone's independent assistant. */
internal class RemotePcControls(private val activity: Activity) : Closeable {
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "orion-pc-controls").apply { isDaemon = true }
    }
    private val client = RemotePcPowerClient(activity)
    private var sheet: BottomSheetDialog? = null
    private var confirmationDialog: androidx.appcompat.app.AlertDialog? = null
    private var pairDialog: androidx.appcompat.app.AlertDialog? = null
    private lateinit var status: TextView
    private lateinit var shutdown: Button
    private lateinit var cancel: Button
    private lateinit var refresh: Button
    private lateinit var connect: Button
    private var pending: JSONObject? = null
    private var available = false
    private var busy = false
    @Volatile private var closed = false
    private val poll = Runnable { if (!closed && !busy) refresh() }
    private fun dp(value: Int) = (value * activity.resources.displayMetrics.density).toInt()

    fun show() {
        if (closed || activity.isFinishing || activity.isDestroyed) return
        val panel = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(22), dp(22), dp(22), dp(24))
            background = GradientDrawable().apply {
                setColor(Color.rgb(17, 17, 20)); cornerRadius = dp(24).toFloat()
                setStroke(dp(1), Color.rgb(59, 43, 48))
            }
        }
        panel.addView(TextView(activity).apply {
            text = "PC CONTROLS"; textSize = 19f; setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            letterSpacing = 0.09f
        })
        status = TextView(activity).apply {
            text = "Connecting to your paired PC…"; textSize = 14f
            setTextColor(Color.rgb(187, 187, 195)); setPadding(0, dp(18), 0, dp(18))
        }
        panel.addView(status)
        fun button(label: String, action: () -> Unit): Button = Button(activity).apply {
            text = label; isAllCaps = false; setTextColor(Color.WHITE)
            backgroundTintList = android.content.res.ColorStateList.valueOf(Color.rgb(70, 29, 38))
            setOnClickListener { action() }
            panel.addView(this, LinearLayout.LayoutParams(-1, dp(52)).apply { topMargin = dp(8) })
        }
        shutdown = button("Shut down PC", ::prepare)
        cancel = button("Cancel shutdown", ::cancel).apply { visibility = android.view.View.GONE }
        refresh = button("Refresh connection", ::refresh)
        connect = button("Pair or change PC", ::pair)
        sheet = BottomSheetDialog(activity).also { dialog ->
            dialog.setContentView(panel)
            dialog.setOnDismissListener { close() }
            dialog.setOnShowListener {
                dialog.findViewById<android.view.View>(com.google.android.material.R.id.design_bottom_sheet)
                    ?.setBackgroundColor(Color.TRANSPARENT)
            }
            dialog.show()
        }
        refresh()
    }

    private fun controls() {
        if (!::shutdown.isInitialized) return
        val state = pending?.optString("state").orEmpty()
        shutdown.isEnabled = !busy && available && PcPowerPresentation.canSchedule(state)
        refresh.isEnabled = !busy
        connect.isEnabled = !busy
        cancel.visibility = if (PcPowerPresentation.canCancel(state)) android.view.View.VISIBLE else android.view.View.GONE
        cancel.isEnabled = !busy && available
    }

    private fun task(action: () -> JSONObject, completed: (JSONObject) -> Unit) {
        if (closed || busy) return
        main.removeCallbacks(poll)
        busy = true; controls()
        worker.execute {
            val outcome = runCatching(action)
            main.post {
                if (closed || activity.isFinishing || activity.isDestroyed) return@post
                busy = false
                outcome.fold(completed) { error ->
                    available = false
                    status.text = if (error is java.io.IOException) error.message
                        else "The PC request could not be confirmed. Refresh the connection before trying again."
                }
                controls()
            }
        }
    }

    private fun render(data: JSONObject) {
        available = data.optBoolean("available")
        pending = data.optJSONObject("pending")
        val value = pending
        status.text = when {
            !available -> "PC shutdown requires a running Windows ORION host."
            value == null -> "PC connected"
            else -> PcPowerPresentation.label(value.optString("state"), value.optInt("remaining_s")) +
                if (value.optString("state") in setOf("blocked", "uncertain", "initiated"))
                    "\n${value.optString("message")}" else ""
        }
        if (value?.optString("state") in setOf("scheduled", "executing")) main.postDelayed(poll, 5_000)
    }

    private fun refresh() { task(client::status, ::render) }

    private fun prepare() {
        task(client::prepare) { challenge ->
            val token = challenge.optString("confirmation")
            if (token.isBlank()) { status.text = "The PC could not prepare a confirmation."; return@task }
            confirmationDialog = MaterialAlertDialogBuilder(activity).setTitle("Shut down your PC?")
                .setMessage("Save your work first. ORION will wait 60 seconds so you can cancel. Windows will not be forced to close apps; unsaved work can block shutdown. ORION must stay running on the PC during the countdown.")
                .setPositiveButton("Shut down PC") { _, _ ->
                    task({ client.shutdown(token) }) { result ->
                        pending = result
                        status.text = PcPowerPresentation.label(result.optString("state"), result.optInt("remaining_s"))
                        main.postDelayed(poll, 1_000)
                    }
                }.setNegativeButton("Keep running", null).create().also { dialog ->
                    dialog.setOnDismissListener { confirmationDialog = null }
                    dialog.show()
                }
        }
    }

    private fun cancel() {
        val key = pending?.optString("request_id")?.takeIf { it.isNotBlank() } ?: return
        task({ client.cancel(key) }) { result ->
            pending = result
            status.text = "PC shutdown cancelled."
        }
    }

    private fun pair() {
        val fields = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL; setPadding(dp(24), dp(8), dp(24), 0)
        }
        val address = EditText(activity).apply {
            hint = "PC HTTPS address"; inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(Prefs.baseUrl(activity).orEmpty()); fields.addView(this)
        }
        val code = EditText(activity).apply {
            hint = "Pairing code"; inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
            fields.addView(this)
        }
        val certificate = EditText(activity).apply {
            hint = "Certificate code (home Wi-Fi address only)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS
            fields.addView(this)
        }
        pairDialog = MaterialAlertDialogBuilder(activity).setTitle("Pair your PC")
            .setMessage("On the same Wi-Fi as the PC, use the https://192.168… address and both codes shown in SECURITY → PAIR PHONE; no Tailscale is needed. Away from home, use the PC's Tailscale or tunnel HTTPS address. Your phone assistant stays independent.")
            .setView(fields).setPositiveButton("Pair") { _, _ ->
                task({ client.pair(address.text.toString(), code.text.toString(), certificate.text.toString()); client.status() }, ::render)
            }.setNegativeButton("Cancel", null).create().also { dialog ->
                dialog.setOnDismissListener { pairDialog = null }
                dialog.show()
            }
    }

    override fun close() {
        if (closed) return
        closed = true
        main.removeCallbacks(poll)
        confirmationDialog?.dismiss(); pairDialog?.dismiss()
        sheet?.dismiss()
        client.close()
        worker.shutdownNow()
    }
}
