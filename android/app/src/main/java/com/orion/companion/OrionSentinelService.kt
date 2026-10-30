package com.orion.companion

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.content.res.ColorStateList
import android.content.res.Configuration
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.Point
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.provider.Settings
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewConfiguration
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import kotlin.math.abs

/** User-enabled, draggable phone controls with an explicit foreground hand-off. */
class OrionSentinelService : Service() {
    private var manager: WindowManager? = null
    private var overlay: FrameLayout? = null
    private var waveform: TactileWaveformView? = null
    private var params: WindowManager.LayoutParams? = null
    private var menu: ScrollView? = null
    private var statusLabel: TextView? = null
    private var listenButton: Button? = null
    private var sharingButton: Button? = null
    private var speechButton: Button? = null
    private var lastState = ""
    private val handler = Handler(Looper.getMainLooper())

    private val refresh = object : Runnable {
        override fun run() {
            if (!running) return
            if (!Settings.canDrawOverlays(this@OrionSentinelService)) {
                stopSelf()
                return
            }
            updateControls()
            handler.postDelayed(this, 750L)
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createChannel()
        try {
            val notification = buildNotification()
            if (Build.VERSION.SDK_INT >= 34) {
                startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
            } else startForeground(NOTIFICATION_ID, notification)
            running = true
            handler.post(refresh)
        } catch (_: RuntimeException) {
            // A revoked permission must not leave a notification or phantom control.
            running = false
            stopSelf()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopSelf(); return START_NOT_STICKY }
            ACTION_STATE -> Unit // Read current service state below rather than retaining stale flags.
        }
        if (!running) return START_NOT_STICKY
        if (overlay == null) attachOverlay()
        updateControls()
        return START_NOT_STICKY
    }

    private fun attachOverlay() {
        if (!Settings.canDrawOverlays(this)) { stopSelf(); return }
        val wm = getSystemService(WINDOW_SERVICE) as WindowManager
        manager = wm
        val face = TactileWaveformView(this).apply {
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        val root = FrameLayout(this).apply {
            contentDescription = "ORION Sentinel. Tap for phone controls; drag to move."
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            isClickable = true
            isFocusable = true
            addView(face, FrameLayout.LayoutParams(dp(72), dp(72)))
            setOnClickListener { if (menu == null) showMenu() else dismissMenu() }
        }
        val lp = overlayParams(dp(72), dp(72)).apply { x = dp(12); y = dp(120) }
        val initial = SentinelLayout.clamp(lp.x, lp.y, lp.width, lp.height, displayBounds())
        lp.x = initial.x; lp.y = initial.y
        var downX = 0f
        var downY = 0f
        var startX = 0
        var startY = 0
        var moved = false
        val slop = ViewConfiguration.get(this).scaledTouchSlop
        root.setOnTouchListener { _, event ->
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    downX = event.rawX; downY = event.rawY
                    startX = lp.x; startY = lp.y; moved = false
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = event.rawX - downX
                    val dy = event.rawY - downY
                    if (abs(dx) > slop || abs(dy) > slop) {
                        moved = true
                        dismissMenu()
                    }
                    if (moved) {
                        val position = SentinelLayout.clamp(
                            (startX + dx).toInt(), (startY + dy).toInt(),
                            lp.width, lp.height, displayBounds(),
                        )
                        lp.x = position.x; lp.y = position.y
                        try { wm.updateViewLayout(root, lp) }
                        catch (_: RuntimeException) { stopSelf() }
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    if (!moved) root.performClick()
                    true
                }
                MotionEvent.ACTION_CANCEL -> { moved = false; true }
                else -> false
            }
        }
        try {
            wm.addView(root, lp)
            overlay = root; waveform = face; params = lp
        } catch (_: RuntimeException) { stopSelf() }
    }

    private fun showMenu() {
        val wm = manager ?: return
        val orb = params ?: return
        if (!Settings.canDrawOverlays(this)) { stopSelf(); return }
        val bounds = displayBounds()
        val width = minOf(dp(304), bounds.width)
        val height = minOf(dp(400), bounds.height)
        if (width <= 0 || height <= 0) return
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(12))
        }
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        header.addView(TextView(this).apply {
            text = "SENTINEL"
            setTextColor(Color.WHITE)
            textSize = 15f
            gravity = Gravity.CENTER_VERTICAL
        }, LinearLayout.LayoutParams(0, dp(48), 1f))
        header.addView(actionButton("Close menu", "Close the Sentinel menu") { dismissMenu() },
            LinearLayout.LayoutParams(dp(104), dp(48)))
        content.addView(header)
        statusLabel = TextView(this).apply {
            setTextColor(Color.rgb(199, 202, 211))
            textSize = 12f
            setPadding(0, dp(4), 0, dp(10))
        }
        content.addView(statusLabel)

        fun pair(first: Button, second: Button) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(first, LinearLayout.LayoutParams(0, dp(52), 1f).apply {
                setMargins(0, dp(3), dp(3), dp(3))
            })
            row.addView(second, LinearLayout.LayoutParams(0, dp(52), 1f).apply {
                setMargins(dp(3), dp(3), 0, dp(3))
            })
            content.addView(row)
        }
        listenButton = actionButton("Start listening", "Toggle background wake-word listening") {
            foreground(SentinelActionContract.TOGGLE_LISTENING)
        }
        pair(actionButton("Talk to ORION", "Open the microphone to give a phone command") {
            foreground(SentinelActionContract.DICTATE)
        }, listenButton!!)
        speechButton = actionButton("Stop speaking", "Stop ORION speech on the phone") {
            RemoteSpeech.stopAll()
            PhoneVoice.stopAll()
            feedback("Stopping speech…")
            updateControls()
        }
        sharingButton = actionButton("Share screen", "Toggle phone screen sharing; Android consent is required") {
            foreground(SentinelActionContract.SCREEN_SHARING)
        }
        pair(speechButton!!, sharingButton!!)
        pair(actionButton("Phone controls", "Open native phone controls and permission settings") {
            foreground(SentinelActionContract.PHONE_CONTROLS)
        }, actionButton("Open ORION", "Return to ORION") {
            foreground(SentinelActionContract.OPEN_ORION)
        })
        pair(actionButton("Back", "Go back on the phone using enabled accessibility controls") {
            navigate("back")
        }, actionButton("Home", "Go to the phone home screen using enabled accessibility controls") {
            navigate("home")
        })
        content.addView(actionButton("Hide Sentinel", "Stop the floating Sentinel and its notification") {
            stopSelf()
        }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(52)).apply {
            topMargin = dp(3)
        })

        val panel = ScrollView(this).apply {
            contentDescription = "ORION phone action menu"
            isFillViewport = false
            clipToOutline = true
            background = shape(Color.rgb(22, 24, 31), Color.rgb(65, 68, 80), 18)
            elevation = dp(12).toFloat()
            addView(content)
            setOnTouchListener { _, event ->
                if (event.actionMasked == MotionEvent.ACTION_OUTSIDE) {
                    dismissMenu(); true
                } else false
            }
        }
        val position = SentinelLayout.menu(
            orb.x, orb.y, orb.width, width, height, dp(6), bounds,
        )
        val lp = overlayParams(width, height).apply {
            flags = flags or WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH
            x = position.x; y = position.y
        }
        try {
            wm.addView(panel, lp)
            menu = panel
            updateControls()
            panel.announceForAccessibility("Sentinel phone controls")
        } catch (_: RuntimeException) {
            dismissMenu()
            if (!Settings.canDrawOverlays(this)) stopSelf()
            else feedback("The phone could not open Sentinel controls")
        }
    }

    private fun actionButton(label: String, description: String, action: () -> Unit) = Button(this).apply {
        text = label
        contentDescription = description
        textSize = 12f
        isAllCaps = false
        setTextColor(Color.WHITE)
        setPadding(dp(6), dp(4), dp(6), dp(4))
        minHeight = dp(48)
        minimumHeight = dp(48)
        minWidth = 0
        minimumWidth = 0
        background = RippleDrawable(ColorStateList.valueOf(Color.rgb(96, 76, 91)),
            shape(Color.rgb(37, 40, 50), Color.rgb(64, 68, 80), 10), null)
        setOnClickListener { action() }
    }

    private fun foreground(action: String) {
        dismissMenu()
        try {
            startActivity(foregroundIntent(action)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP))
        } catch (_: RuntimeException) {
            feedback("Could not open ORION. Open the app to continue.")
        }
    }

    private fun navigate(action: String) {
        dismissMenu()
        if (!OrionAccessibilityService.isConnected()) {
            feedback("Enable ORION accessibility to use $action")
            foreground(SentinelActionContract.PHONE_CONTROLS)
            return
        }
        // The tap itself authorises this local navigation; no remote command is sent.
        OrionAccessibilityService.navigate(action) { result ->
            if (!result.optBoolean("ok")) feedback("Android could not perform $action")
        }
    }

    private fun updateControls() {
        val listening = PhoneListeningService.running || OrionAudioService.running
        val sharing = OrionProjectionService.isCapturing
        val voice = RemoteSpeech.anySpeaking() || PhoneVoice.anySpeaking()
        waveform?.watching = sharing || listening
        waveform?.speaking = voice
        listenButton?.text = if (listening) "Stop listening" else "Start listening"
        listenButton?.contentDescription = if (listening) "Stop background listening" else "Start background listening"
        sharingButton?.text = if (sharing) "Stop sharing" else "Share screen"
        sharingButton?.contentDescription = if (sharing) "Stop phone screen sharing" else "Share the phone screen with Android consent"
        speechButton?.isEnabled = voice
        speechButton?.alpha = if (voice) 1f else 0.55f
        val status = buildString {
            append(if (listening) "Listening for ORION" else "Listening is off")
            append(if (sharing) " · Sharing screen" else " · Screen sharing off")
            if (voice) append("\nORION is speaking")
            if (!OrionAccessibilityService.isConnected()) append("\nBack and Home need accessibility")
        }
        statusLabel?.text = status
        if (status != lastState) {
            lastState = status
            if (running) (getSystemService(NOTIFICATION_SERVICE) as NotificationManager)
                .notify(NOTIFICATION_ID, buildNotification())
        }
    }

    private fun dismissMenu() {
        menu?.let { try { manager?.removeView(it) } catch (_: RuntimeException) {} }
        menu = null; statusLabel = null; listenButton = null; sharingButton = null; speechButton = null
    }

    private fun overlayParams(width: Int, height: Int) = WindowManager.LayoutParams(
        width, height, WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
        PixelFormat.TRANSLUCENT,
    ).apply { gravity = Gravity.TOP or Gravity.START }

    private fun displayBounds(): SentinelLayout.Bounds {
        val wm = manager ?: getSystemService(WINDOW_SERVICE) as WindowManager
        val margin = dp(6)
        if (Build.VERSION.SDK_INT >= 30) {
            val metrics = wm.currentWindowMetrics
            val rect = metrics.bounds
            val inset = metrics.windowInsets.getInsetsIgnoringVisibility(
                WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout(),
            )
            return SentinelLayout.Bounds(
                inset.left + margin, inset.top + margin,
                rect.width() - inset.right - margin, rect.height() - inset.bottom - margin,
            )
        }
        val screen = Point()
        @Suppress("DEPRECATION")
        wm.defaultDisplay.getRealSize(screen)
        // API 26–29 do not provide WindowMetrics. Keep the menu away from system bars.
        return SentinelLayout.Bounds(margin, dp(28) + margin, screen.x - margin, screen.y - dp(48) - margin)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        dismissMenu()
        val lp = params ?: return
        val root = overlay ?: return
        val position = SentinelLayout.clamp(lp.x, lp.y, lp.width, lp.height, displayBounds())
        lp.x = position.x; lp.y = position.y
        try { manager?.updateViewLayout(root, lp) } catch (_: RuntimeException) { stopSelf() }
    }

    private fun shape(fill: Int, border: Int, radiusDp: Int) = GradientDrawable().apply {
        setColor(fill)
        setStroke(dp(1).coerceAtLeast(1), border)
        cornerRadius = dp(radiusDp).toFloat()
    }

    private fun createChannel() {
        val notifications = getSystemService(NOTIFICATION_SERVICE) as NotificationManager
        notifications.createNotificationChannel(NotificationChannel(
            CHANNEL_ID, "ORION Sentinel", NotificationManager.IMPORTANCE_LOW))
    }

    private fun activityAction(request: Int, action: String) = PendingIntent.getActivity(
        this, request, foregroundIntent(action)
            .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
    )

    private fun foregroundIntent(action: String): Intent {
        val phoneMode = PhoneAiProfile.isPhoneMode(this)
        val target = if (phoneMode) PhoneConsoleActivity::class.java else MainActivity::class.java
        return Intent(this, target)
            .putExtra(SentinelActionContract.EXTRA_ACTION, action)
            .putExtra("desktop_mode", !phoneMode)
    }

    private fun buildNotification(): Notification {
        val open = activityAction(0, SentinelActionContract.OPEN_ORION)
        val talk = activityAction(2, SentinelActionContract.DICTATE)
        val listen = activityAction(3, SentinelActionContract.TOGGLE_LISTENING)
        val stop = PendingIntent.getService(this, 1,
            Intent(this, OrionSentinelService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_orion)
            .setContentTitle("ORION Sentinel")
            .setContentText(if (PhoneListeningService.running || OrionAudioService.running)
                "Listening for ORION · Tap the ring for actions" else "Tap the ring for phone actions")
            .setContentIntent(open)
            .addAction(android.R.drawable.ic_btn_speak_now, "Talk", talk)
            .addAction(android.R.drawable.ic_media_play,
                if (PhoneListeningService.running || OrionAudioService.running) "Stop listening" else "Listen", listen)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Hide", stop)
            .setOngoing(true)
            .build()
    }

    private fun feedback(message: String) = Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    override fun onDestroy() {
        running = false
        handler.removeCallbacksAndMessages(null)
        dismissMenu()
        overlay?.let { try { manager?.removeView(it) } catch (_: RuntimeException) {} }
        overlay = null; waveform = null; manager = null; params = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    companion object {
        private const val CHANNEL_ID = "orion_sentinel"
        private const val NOTIFICATION_ID = 7602
        private const val ACTION_STOP = "com.orion.companion.SENTINEL_STOP"
        private const val ACTION_STATE = "com.orion.companion.SENTINEL_STATE"
        private const val EXTRA_WATCHING = "watching"
        private const val EXTRA_SPEAKING = "speaking"
        @Volatile var running = false
            private set

        fun start(context: Context) {
            context.startForegroundService(Intent(context, OrionSentinelService::class.java))
        }

        fun stop(context: Context) {
            context.stopService(Intent(context, OrionSentinelService::class.java))
        }

        fun update(context: Context, watching: Boolean, speaking: Boolean) {
            if (!running) return
            context.startService(Intent(context, OrionSentinelService::class.java)
                .setAction(ACTION_STATE)
                .putExtra(EXTRA_WATCHING, watching)
                .putExtra(EXTRA_SPEAKING, speaking))
        }
    }
}
