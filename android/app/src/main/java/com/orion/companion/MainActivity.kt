package com.orion.companion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.media.projection.MediaProjectionManager
import android.net.ConnectivityManager
import android.net.Network
import android.net.Uri
import android.net.MailTo
import android.net.http.SslError
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.InputType
import android.view.MotionEvent
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.JsPromptResult
import android.webkit.PermissionRequest
import android.webkit.RenderProcessGoneDetail
import android.webkit.SslErrorHandler
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import android.widget.PopupMenu
import android.widget.ScrollView
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.webkit.ScriptHandler
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.orion.companion.databinding.ActivityMainBinding
import org.json.JSONObject
import org.json.JSONTokener
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.atomic.AtomicIntegerArray
import kotlin.math.min
import kotlin.random.Random

/**
 * The native shell: a full-screen WebView onto ORION's uplink. His voxel
 * orb/face, live emotion mirror, spoken replies and memory all come from the
 * uplink the desktop already serves — this app makes it a first-class Android
 * app: native config, microphone permission, HTTPS validation, keep-
 * awake and immersive full-screen.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var pcControls: RemotePcControls? = null

    /** The endpoints that answered the last reachability probe, best first —
     *  what the WebView loads and fails over through. */
    private var candidates: List<String> = emptyList()
    private var candidateIndex = 0
    private var trustedOrigins: Set<String> = emptySet()
    @Volatile private var trustedPageActive = false

    /** Bumped by every connect(), so a slow probe from an earlier attempt can't
     *  load over a newer one. */
    private var connectAttempt = 0
    private var loadFailed = false
    private var resumedOnce = false
    private var webViewGone = false
    private var touchDownRawY = 0f
    private var credentialScript: ScriptHandler? = null
    private var pendingMicRequest: PermissionRequest? = null
    private lateinit var intentRouter: IntentRouter
    private var resumed = false
    private var uplinkConnected = false
    private var pingMs: Long? = null
    private var modelRoute = "--"
    private var micActive = false
    private var speakingActive = false
    private var phoneSpeech: RemoteSpeech? = null
    private var phoneDictation: NativeDictation? = null
    private var lastAudioStatus = ""
    private var microphoneSwitch = 0
    private var lastSentWatching: Boolean? = null
    private var lastSentSpeaking: Boolean? = null
    private var pendingPhonePermission: ((Boolean) -> Unit)? = null
    private var afterNotificationPrompt: (() -> Unit)? = null
    private var activeAuthorization: Authorization? = null
    private var authorizationExpiry: Runnable? = null
    private var authorizationOverlay: View? = null
    private var approvedSnapshotId: String? = null
    private var approvedSnapshot: JSONObject? = null
    private var pendingSentinelAction: String? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private var retryAttempt = 0
    private var retryConnection: Runnable? = null
    private var pageLoadTimeout: Runnable? = null
    private var networkReconnect: Runnable? = null
    private var networkCallbackRegistered = false
    private val networkCallback = object : ConnectivityManager.NetworkCallback() {
        override fun onAvailable(network: Network) = queueNetworkReconnect()
        override fun onLost(network: Network) = queueNetworkReconnect()
    }

    private fun queueNetworkReconnect() {
        mainHandler.post {
            networkReconnect?.let(mainHandler::removeCallbacks)
            networkReconnect = Runnable {
                if (!isFinishing && !isDestroyed && ::binding.isInitialized) {
                    OrionFrameTransport.stop()
                    if (Prefs.credentials(this) != null) OrionFrameTransport.start(this)
                    connect()
                }
            }.also { mainHandler.postDelayed(it, 1200) }
        }
    }
    private val telemetryTicker = object : Runnable {
        override fun run() {
            if (!resumed || isDestroyed) return
            updateTelemetry()
            mainHandler.postDelayed(this, 200)
        }
    }

    private data class Authorization(
        val details: String,
        val approve: () -> Unit,
        val deny: () -> Unit,
        val commandId: String? = null,
    )

    private val phonePermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            pendingPhonePermission?.let { callback ->
                pendingPhonePermission = null
                callback(granted)
            }
        }

    private val notificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) {
            afterNotificationPrompt?.invoke()
            afterNotificationPrompt = null
        }

    private val projectionConsent =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            result.data?.let { data ->
                if (result.resultCode == RESULT_OK) {
                    OrionProjectionService.start(this, result.resultCode, data)
                }
            }
            updateTelemetry()
        }

    private fun withNotificationPermission(start: () -> Unit) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            afterNotificationPrompt = start
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else start()
    }

    private val overlayPermission =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) {
            if (Settings.canDrawOverlays(this)) {
                withNotificationPermission {
                    lastSentWatching = null
                    OrionSentinelService.start(this)
                }
            }
            updateTelemetry()
        }

    /** Ask for the microphone when ORION's page first wants it (the mic button),
     *  not at launch — and ask again later if it was refused the first time. */
    private val micPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            val request = pendingMicRequest ?: return@registerForActivityResult
            pendingMicRequest = null
            if (granted) {
                request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
            } else {
                request.deny()
                Toast.makeText(this, R.string.mic_denied, Toast.LENGTH_LONG).show()
            }
        }

    private val backgroundMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted && Prefs.credentials(this) != null) startBackgroundListening()
            else Toast.makeText(this, R.string.mic_denied, Toast.LENGTH_LONG).show()
            updateTelemetry()
        }

    private val dictationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted && resumed) phoneDictation?.start()
            else dictationEvent(JSONObject().put("state", "error")
                .put("text", "Allow ORION's microphone permission in Android settings."))
        }

    private fun dictationEvent(event: JSONObject) {
        if (!::binding.isInitialized || webViewGone || !isTrustedOrigin(binding.webView.url)) return
        when (event.optString("state")) {
            "listening" -> micActive = true
            "transcribing", "finished", "error", "transcript" -> micActive = false
        }
        updateTelemetry()
        binding.webView.evaluateJavascript("window.onNativeDictation&&window.onNativeDictation($event)", null)
    }

    private fun startPhoneDictation() {
        if (!resumed || !isTrustedOrigin(binding.webView.url)) return
        val wasListening = OrionAudioService.running
        OrionAudioService.stop(this)
        phoneSpeech?.stop()
        if (phoneDictation == null) phoneDictation = NativeDictation(this, ::dictationEvent)
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED) {
            if (wasListening) waitForMicrophoneRelease { phoneDictation?.start() }
            else phoneDictation?.start()
        } else dictationPermission.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun waitForMicrophoneRelease(ready: () -> Unit) {
        val attempt = ++microphoneSwitch
        val deadline = android.os.SystemClock.elapsedRealtime() + 2500
        fun poll() {
            if (attempt != microphoneSwitch || !resumed || isDestroyed) return
            if (!OrionAudioService.captureActive && phoneDictation?.busy != true) ready()
            else if (android.os.SystemClock.elapsedRealtime() < deadline) mainHandler.postDelayed(::poll, 40)
            else dictationEvent(JSONObject().put("state", "error")
                .put("text", "The previous microphone session has not stopped. Please try again."))
        }
        poll()
    }

    private fun startBackgroundListening() {
        if (!resumed || !isTrustedOrigin(binding.webView.url)) return
        phoneDictation?.cancel()
        micActive = false
        phoneSpeech?.stop()
        withNotificationPermission {
            waitForMicrophoneRelease { OrionAudioService.start(this) }
        }
    }

    private fun showOptions() {
        PopupMenu(this, binding.settingsButton).apply {
            menu.add(0, 1, 0, if (OrionAudioService.running) "Stop background listening" else "Start background listening")
            menu.add(0, 2, 1, "PC statistics")
            menu.add(0, 3, 2, "Tasks")
            menu.add(0, 4, 3, if (OrionProjectionService.isCapturing) "Stop screen sharing" else "Share phone screen")
            menu.add(0, 5, 4, "Phone accessibility")
            menu.add(0, 6, 5, if (OrionSentinelService.running) "Stop Sentinel overlay" else "Start Sentinel overlay")
            menu.add(0, 7, 6, "Connection & microphone")
            menu.add(0, 8, 7, getString(R.string.change_server))
            menu.add(0, 9, 8, "Stop speaking")
            menu.add(0, 10, 9, "Phone controls")
            menu.add(0, 11, 10, "Use ORION on this phone")
            menu.add(0, 12, 11, "PC controls")
            setOnMenuItemClickListener { item ->
                when (item.itemId) {
                    1 -> binding.audioButton.performClick()
                    2 -> binding.webView.evaluateJavascript("window.openOrionHardware&&window.openOrionHardware()", null)
                    3 -> binding.webView.evaluateJavascript("window.openOrionTasks&&window.openOrionTasks()", null)
                    4 -> binding.visionButton.performClick()
                    5 -> binding.accessButton.performClick()
                    6 -> binding.sentinelButton.performClick()
                    7 -> binding.webView.evaluateJavascript("window.openOrionDiagnostics&&window.openOrionDiagnostics()", null)
                    8 -> openSetup()
                    9 -> RemoteSpeech.stopAll()
                    10 -> PhoneControls.show(this@MainActivity, ::routeNativeAction)
                    11 -> {
                        PhoneAiProfile.setPhoneMode(this@MainActivity, true)
                        startActivity(Intent(this@MainActivity, PhoneConsoleActivity::class.java))
                        finish()
                    }
                    12 -> {
                        pcControls?.close()
                        pcControls = RemotePcControls(this@MainActivity).also { it.show() }
                    }
                }
                true
            }
            show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (PhoneAiProfile.isPhoneMode(this) && !intent.getBooleanExtra("desktop_mode", false)) {
            startActivity(Intent(this, PhoneConsoleActivity::class.java).apply {
                intent.getStringExtra(SentinelActionContract.EXTRA_ACTION)?.let {
                    putExtra(SentinelActionContract.EXTRA_ACTION, it)
                }
            })
            finish()
            return
        }

        if (Prefs.candidateUrls(this).isEmpty()) {
            startActivity(Intent(this, SetupActivity::class.java))
            finish()
            return
        }
        trustedOrigins = Prefs.candidateUrls(this).mapNotNull(::originOf).toSet()

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        makeApprovalScrollable()
        pendingSentinelAction = intent.getStringExtra(SentinelActionContract.EXTRA_ACTION)
            ?.takeIf(SentinelActionContract::isSupported)
        phoneSpeech = RemoteSpeech(this, onSpeaking = { active ->
            if (!isDestroyed && !webViewGone) {
                speakingActive = active
                updateTelemetry()
                if (isTrustedOrigin(binding.webView.url)) {
                    binding.webView.evaluateJavascript(
                        "if(window.orionNativeSpeechState)orionNativeSpeechState($active);", null)
                }
            }
        }, onVoice = { voice ->
            if (voice.startsWith("Voice "))
                Toast.makeText(this, voice, Toast.LENGTH_LONG).show()
        })
        intentRouter = IntentRouter(this)
        installTransportListener()

        applyImmersiveKeepAwake()
        applyInsets()
        configureWebView()
        configurePullToRefresh()
        installCredentialSeed()
        try {
            (getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager)
                .registerDefaultNetworkCallback(networkCallback)
            networkCallbackRegistered = true
        } catch (_: Exception) { }

        binding.settingsButton.setOnClickListener { showOptions() }
        binding.changeServerButton.setOnClickListener { openSetup() }
        binding.connectingChangeServerButton.setOnClickListener { openSetup() }
        binding.retryButton.setOnClickListener { connect() }
        binding.visionButton.setOnClickListener {
            if (OrionProjectionService.isCapturing) {
                OrionProjectionService.stop(this)
            } else {
                withNotificationPermission {
                    val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
                    projectionConsent.launch(manager.createScreenCaptureIntent())
                }
            }
            updateTelemetry()
        }
        binding.audioButton.setOnClickListener {
            if (OrionAudioService.running) OrionAudioService.stop(this)
            else if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED) {
                if (Prefs.credentials(this) != null) startBackgroundListening()
                else Toast.makeText(this, R.string.pair_message, Toast.LENGTH_LONG).show()
            } else backgroundMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            updateTelemetry()
        }
        binding.accessButton.setOnClickListener {
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
        }
        binding.sentinelButton.setOnClickListener {
            if (OrionSentinelService.running) OrionSentinelService.stop(this)
            else if (Settings.canDrawOverlays(this)) withNotificationPermission {
                lastSentWatching = null
                OrionSentinelService.start(this)
            }
            else overlayPermission.launch(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:$packageName")))
            updateTelemetry()
        }
        binding.authorizationDeny.setOnClickListener { finishAuthorization(false) }
        updateTelemetry()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                // Leaving keeps ORION's page — and the conversation on it — alive,
                // as Home does (and as Android 12+ treats a launcher activity),
                // rather than destroying it and starting over next time.
                if (!webViewGone && binding.webView.canGoBack()) binding.webView.goBack()
                else moveTaskToBack(true)
            }
        })

        connect()
    }

    /** Back from "Change server": reconnect with whatever was just saved. */
    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        if (PhoneAiProfile.isPhoneMode(this) && !intent.getBooleanExtra("desktop_mode", false)) {
            startActivity(Intent(this, PhoneConsoleActivity::class.java).apply {
                intent.getStringExtra(SentinelActionContract.EXTRA_ACTION)?.let {
                    putExtra(SentinelActionContract.EXTRA_ACTION, it)
                }
            })
            finish()
            return
        }
        pendingSentinelAction = intent.getStringExtra(SentinelActionContract.EXTRA_ACTION)
            ?.takeIf(SentinelActionContract::isSupported)
        if (resumed) handleSentinelAction()
        if (intent.getBooleanExtra(EXTRA_RECONFIGURED, false) && ::binding.isInitialized) {
            OrionFrameTransport.stop()
            trustedOrigins = Prefs.candidateUrls(this).mapNotNull(::originOf).toSet()
            installCredentialSeed()
            connect()
            if (Prefs.credentials(this) != null) OrionFrameTransport.start(this)
        }
    }

    override fun onResume() {
        super.onResume()
        resumed = true
        if (!::binding.isInitialized || webViewGone) return
        mainHandler.removeCallbacks(telemetryTicker)
        mainHandler.post(telemetryTicker)
        activeAuthorization?.let { pending ->
            if (authorizationOverlay != null) {
                removeAuthorizationOverlay()
                showForegroundAuthorization(pending)
            }
        }
        binding.webView.onResume()
        if (resumedOnce) recheckConnection() else resumedOnce = true
        if (Prefs.credentials(this) != null) OrionFrameTransport.start(this)
        updateTelemetry()
        handleSentinelAction()
    }

    private fun handleSentinelAction() {
        if (!resumed || !::binding.isInitialized) return
        val action = pendingSentinelAction ?: return
        pendingSentinelAction = null
        when (action) {
            SentinelActionContract.DICTATE -> {
                if (isTrustedOrigin(binding.webView.url)) startPhoneDictation()
                else Toast.makeText(this, "Connect the desktop console or open ORION on this phone before dictation", Toast.LENGTH_LONG).show()
            }
            SentinelActionContract.TOGGLE_LISTENING -> binding.audioButton.performClick()
            SentinelActionContract.SCREEN_SHARING -> binding.visionButton.performClick()
            SentinelActionContract.PHONE_CONTROLS -> PhoneControls.show(this, ::routeNativeAction)
            SentinelActionContract.OPEN_ORION -> Unit
        }
    }

    private fun makeApprovalScrollable() {
        val details = binding.authorizationDetails
        val parent = details.parent as? ViewGroup ?: return
        val position = parent.indexOfChild(details)
        parent.removeView(details)
        parent.addView(ScrollView(this).apply { addView(details) }, position,
            ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                (180 * resources.displayMetrics.density).toInt()))
    }

    override fun onPause() {
        resumed = false
        mainHandler.removeCallbacks(telemetryTicker)
        phoneDictation?.finish()
        activeAuthorization?.let { pending ->
            if (authorizationOverlay == null && !isFinishing) {
                binding.authorizationDeck.visibility = View.GONE
                if (Settings.canDrawOverlays(this)) showOverlayAuthorization(pending)
                else finishAuthorization(false)
            }
        }
        if (::binding.isInitialized && !webViewGone) {
            captureCredentials()
            binding.webView.onPause()
        }
        super.onPause()
    }

    // ── connecting ────────────────────────────────────────────────────────────

    /** Probe every known endpoint at once, then load the best one that answers. */
    private fun connect() {
        trustedPageActive = false
        val attempt = ++connectAttempt
        pageLoadTimeout?.let(mainHandler::removeCallbacks)
        pageLoadTimeout = null
        val all = Prefs.candidateUrls(this)
        hideError()
        showConnecting(all.firstOrNull()?.let(::displayAddress).orEmpty())
        Thread {
            var reachable = probe(all)
            if (reachable.isEmpty() && LanDiscovery.rediscover(this) != null) {
                // The PC's LAN address moved; it was found by its pinned
                // certificate on this Wi-Fi. Try again with the new address.
                reachable = probe(Prefs.candidateUrls(this))
            }
            runOnUiThread {
                val origins = Prefs.candidateUrls(this).mapNotNull(::originOf).toSet()
                if (origins != trustedOrigins && !isDestroyed) { trustedOrigins = origins; installCredentialSeed() }
                if (attempt != connectAttempt || isFinishing || isDestroyed) return@runOnUiThread
                if (reachable.isEmpty()) {
                    showError(getString(
                        R.string.error_no_answer, all.joinToString("\n") { displayAddress(it) }))
                } else {
                    candidates = reachable
                    candidateIndex = 0
                    loadCurrent()
                }
            }
        }.apply { isDaemon = true; name = "orion-connect" }.start()
    }

    /**
     * TCP-connect to every endpoint in parallel and return those that answered,
     * in priority order (Tailscale name → LAN → public HTTPS). Without this an
     * unreachable LAN address — you're out on mobile data — left the WebView on a
     * black screen for the whole TCP timeout (minutes) before failing over.
     * Returns as soon as the best endpoint that answered has nothing better still
     * pending; any still unresolved then are kept last as a fallback.
     */
    private fun probe(urls: List<String>): List<String> {
        if (urls.isEmpty()) return emptyList()
        val state = AtomicIntegerArray(urls.size)          // 0 pending, 1 up, 2 down
        urls.forEachIndexed { i, url ->
            Thread { state.set(i, if (isReachable(url)) 1 else 2) }
                .apply { isDaemon = true; name = "orion-probe-$i" }.start()
        }
        val deadline = System.currentTimeMillis() + PROBE_TIMEOUT_MS + 300
        while (System.currentTimeMillis() < deadline) {
            val firstUp = urls.indices.firstOrNull { state.get(it) == 1 }
            if (firstUp != null && (0 until firstUp).none { state.get(it) == 0 }) break
            if (urls.indices.none { state.get(it) == 0 }) break
            Thread.sleep(40)
        }
        val up = urls.filterIndexed { i, _ -> state.get(i) == 1 }
        if (up.isEmpty()) return emptyList()
        return up + urls.filterIndexed { i, _ -> state.get(i) == 0 }
    }

    private fun isReachable(url: String): Boolean = try {
        val uri = Uri.parse(url)
        val host = uri.host.orEmpty()
        val port = if (uri.port > 0) uri.port
        else if (uri.scheme.equals("https", ignoreCase = true)) 443 else 80
        if (host.isEmpty()) false else {
            // A mobile network can advertise IPv6 while only its IPv4 path to
            // the tunnel works (or vice versa). Try each resolved address within
            // one bounded probe instead of rejecting the whole hostname after
            // the first failed socket.
            val deadline = android.os.SystemClock.elapsedRealtime() + PROBE_TIMEOUT_MS
            InetAddress.getAllByName(host).any { address ->
                val remaining = deadline - android.os.SystemClock.elapsedRealtime()
                if (remaining <= 0) false else try {
                    Socket().use { socket ->
                        socket.connect(InetSocketAddress(address, port),
                            min(1_500L, remaining).toInt())
                        true
                    }
                } catch (_: Exception) { false }
            }
        }
    } catch (_: Exception) {
        false
    }

    /** Coming back after moving between networks (home Wi-Fi ⇄ mobile data): if
     *  the endpoint the page is on no longer answers, reconnect through the best
     *  one that does, instead of leaving a dead page on screen. */
    private fun recheckConnection() {
        if (binding.errorView.visibility == View.VISIBLE) {
            connect()
            return
        }
        if (binding.connectingView.visibility == View.VISIBLE) return   // already on it
        val current = candidates.getOrNull(candidateIndex) ?: return
        val attempt = connectAttempt
        Thread {
            if (!isReachable(current)) {
                runOnUiThread {
                    if (attempt == connectAttempt && !isFinishing && !isDestroyed) connect()
                }
            }
        }.apply { isDaemon = true; name = "orion-recheck" }.start()
    }

    /** Load the current candidate endpoint (adding ?lite=1 on metered data). */
    private fun loadCurrent() {
        val url = candidates.getOrNull(candidateIndex) ?: return
        trustedPageActive = false
        loadFailed = false
        binding.connectingDetail.text = displayAddress(url)
        pageLoadTimeout?.let(mainHandler::removeCallbacks)
        pageLoadTimeout = Runnable {
            if (isFinishing || isDestroyed || binding.connectingView.visibility != View.VISIBLE ||
                candidates.getOrNull(candidateIndex) != url) return@Runnable
            loadFailed = true
            binding.webView.stopLoading()
            if (candidateIndex < candidates.size - 1) {
                candidateIndex++
                loadCurrent()
            } else {
                showError(getString(R.string.error_load_timeout) + "\n" + displayAddress(url))
            }
        }.also { mainHandler.postDelayed(it, 20_000) }
        binding.webView.loadUrl(withLite(url))
    }

    /** On a metered connection (or when the user forced it) request ORION's
     *  data-light UI, which drops the CDN-backed voxel face. */
    private fun withLite(base: String): String {
        val lite = when (Prefs.litePreference(this)) {
            "on" -> true
            "off" -> false
            else -> isMetered()
        }
        if (!lite) return base
        return if (base.contains("?")) "$base&lite=1" else "$base?lite=1"
    }

    private fun isMetered(): Boolean = try {
        (getSystemService(CONNECTIVITY_SERVICE) as? ConnectivityManager)
            ?.isActiveNetworkMetered ?: false
    } catch (_: Exception) {
        false
    }

    private fun isTrustedOrigin(url: String?): Boolean =
        originOf(url.orEmpty())?.let { candidate ->
            trustedOrigins.any { it.equals(candidate, ignoreCase = true) }
        } == true

    private fun displayAddress(url: String): String = url.substringBefore('?').trimEnd('/')

    private fun openSetup() {
        // Setup sits on top of this screen, so Back returns here unchanged; a
        // new address comes back through onNewIntent().
        startActivity(Intent(this, SetupActivity::class.java))
    }

    /** Hand a non-web URI (tel:, sms:, mailto:, geo:, …) to the phone's app. */
    private fun launchExternal(uri: Uri) {
        when (uri.scheme?.lowercase()) {
            "http", "https" -> routeNativeAction(JSONObject().put("kind", "open_url").put("url", uri.toString()))
            "mailto" -> {
                val mail = runCatching { MailTo.parse(uri.toString()) }.getOrNull()
                if (mail == null || !mail.cc.isNullOrBlank() || !mail.headers["bcc"].isNullOrBlank()) {
                    Toast.makeText(this, "Use a single reviewed e-mail recipient in Phone controls", Toast.LENGTH_LONG).show()
                } else routeNativeAction(JSONObject().put("kind", "compose_email").put("to", mail.to.orEmpty())
                    .put("subject", mail.subject.orEmpty()).put("body", mail.body.orEmpty()))
            }
            "tel" -> routeNativeAction(JSONObject().put("kind", "make_call").put("number", uri.schemeSpecificPart))
            "file", "content", "javascript", "data", "intent", null ->
                Toast.makeText(this, "This link cannot be opened by ORION", Toast.LENGTH_LONG).show()
            else -> requestAuthorization("PHONE APP LINK: $uri\nACTION: Open this exact link in a phone app", {
                try { startActivity(Intent(Intent.ACTION_VIEW, uri)) }
                catch (_: Exception) { Toast.makeText(this, "No app can open this phone link", Toast.LENGTH_LONG).show() }
            }, deny = {})
        }
    }

    // ── pairing that survives switching endpoint ─────────────────────────────

    /**
     * The page keeps its pairing in localStorage, which is per ORIGIN: pairing
     * over the LAN address left the Tailscale address unpaired (and vice versa),
     * so the first time the app connected through the other endpoint it asked
     * for a fresh code — which the desktop only offers on request. Seed the
     * pairing we captured natively into every ORION origin before the page's own
     * script runs, restricted to ORION's own endpoints.
     */
    private fun installCredentialSeed() {
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) return
        credentialScript?.remove()
        credentialScript = null
        if (webViewGone) return
        val (device, refresh) = Prefs.credentials(this) ?: return
        val origins = Prefs.candidateUrls(this).mapNotNull(::originOf).toSet()
        if (origins.isEmpty()) return
        val js = "(function(){try{var s=window.localStorage;" +
            "if(!s.getItem('orion_device')||!s.getItem('orion_refresh')){" +
            "s.setItem('orion_device'," + JSONObject.quote(device) + ");" +
            "s.setItem('orion_refresh'," + JSONObject.quote(refresh) + ");" +
            "s.removeItem('orion_access');s.removeItem('orion_access_exp');}}catch(e){}})();"
        credentialScript = try {
            WebViewCompat.addDocumentStartJavaScript(binding.webView, js, origins)
        } catch (_: Exception) {
            // An origin rule WebView won't accept (unusual IPv6 forms): retry without them.
            try {
                WebViewCompat.addDocumentStartJavaScript(
                    binding.webView, js, origins.filterNot { it.contains('[') }.toSet())
            } catch (_: Exception) {
                null
            }
        }
    }

    private fun originOf(url: String): String? = try {
        val uri = Uri.parse(url)
        val scheme = uri.scheme?.lowercase()
        val host = uri.host.orEmpty().lowercase()
        if (scheme == null || host.isEmpty()) null
        else {
            // WebView strips the default port from its callback URL even when
            // the saved endpoint explicitly says :443. Both spellings are the
            // same origin; comparing them literally kept the loading veil up
            // and could discard the pairing prompt.
            val port = uri.port
            val defaultPort = (scheme == "https" && port == 443) ||
                (scheme == "http" && port == 80)
            "$scheme://$host" + (if (port > 0 && !defaultPort) ":$port" else "")
        }
    } catch (_: Exception) {
        null
    }

    /** Read the pairing the page holds for the current origin and keep a copy. */
    private fun captureCredentials() {
        if (webViewGone) return
        val web = binding.webView
        if (!isTrustedOrigin(web.url)) return
        try {
            web.evaluateJavascript(READ_CREDENTIALS_JS) { raw ->
                val json = try {
                    JSONTokener(raw).nextValue() as? String
                } catch (_: Exception) {
                    null
                } ?: return@evaluateJavascript
                val obj = try {
                    JSONObject(json)
                } catch (_: Exception) {
                    return@evaluateJavascript
                }
                val device = obj.optString("d")
                val refresh = obj.optString("r")
                if (device.isNotEmpty() && refresh.isNotEmpty() &&
                    Prefs.credentials(this) != (device to refresh)
                ) {
                    Prefs.saveCredentials(this, device, refresh)
                    installCredentialSeed()
                    OrionFrameTransport.stop()
                    OrionFrameTransport.start(this)
                }
            }
        } catch (_: Exception) {
        }
    }

    /** The page just paired or refreshed its token: learn the new endpoints and
     *  keep a copy of the pairing for the other endpoints. */
    private fun onEndpointsSaved() {
        if (isFinishing || isDestroyed) return
        trustedOrigins = Prefs.candidateUrls(this).mapNotNull(::originOf).toSet()
        installCredentialSeed()
        captureCredentials()
    }

    private fun installTransportListener() {
        OrionFrameTransport.setListener(object : OrionFrameTransport.Listener {
            override fun onConnectionChanged(connected: Boolean, latencyMs: Long?) {
                runOnUiThread {
                    uplinkConnected = connected
                    if (!connected && activeAuthorization?.commandId != null) {
                        finishAuthorization(false)
                    }
                    pingMs = if (connected) latencyMs ?: pingMs else null
                    val origins = Prefs.candidateUrls(this@MainActivity)
                        .mapNotNull(::originOf).toSet()
                    if (origins != trustedOrigins) {
                        trustedOrigins = origins
                        installCredentialSeed()
                    }
                    updateTelemetry()
                    if (connected) reportPhoneStatus()
                }
            }

            override fun onCommandRequest(request: JSONObject) {
                runOnUiThread { routeDeviceCommand(request) }
            }
        })
    }

    private fun reportPhoneStatus() {
        OrionFrameTransport.sendJson(JSONObject().put("jsonrpc", "2.0")
            .put("method", "device.status").put("params", JSONObject()
                .put("accessibility", OrionAccessibilityService.isConnected())
                .put("screen_capture", OrionProjectionService.isCapturing)
                .put("background_audio", OrionAudioService.running)
                .put("microphone", ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED)
                .put("calls", ContextCompat.checkSelfPermission(this, Manifest.permission.CALL_PHONE) == PackageManager.PERMISSION_GRANTED)
                .put("sms", ContextCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED)
                .put("sentinel", OrionSentinelService.running)
                .put("native_speech", true)))
    }

    private fun updateTelemetry() {
        if (::binding.isInitialized && !webViewGone && isTrustedOrigin(binding.webView.url)) {
            val events = OrionAudioService.drainEvents()
            for (event in events) {
                binding.webView.evaluateJavascript(
                    "window.onBackgroundAudio&&window.onBackgroundAudio($event)", null)
            }
            val status = OrionAudioService.status
            if (events.isNotEmpty()) lastAudioStatus = status
            else if (status != lastAudioStatus) {
                lastAudioStatus = status
                binding.webView.evaluateJavascript("window.onBackgroundAudio&&window.onBackgroundAudio($status)", null)
            }
            if (OrionAudioService.running) {
                val meter = JSONObject().put("state", "level").put("level", OrionAudioService.level)
                binding.webView.evaluateJavascript("window.onBackgroundAudio&&window.onBackgroundAudio($meter)", null)
            }
        }
        if (!::binding.isInitialized || isDestroyed) return
        val access = OrionAccessibilityService.isConnected()
        val watching = OrionProjectionService.isCapturing
        binding.telemetryText.text = if (uplinkConnected) "ORION · Connected" else "ORION · Connecting…"
        binding.visionButton.setText(if (watching) R.string.vision_on else R.string.vision_off)
        binding.audioButton.setText(if (OrionAudioService.running) R.string.audio_on else R.string.audio_off)
        binding.accessButton.setText(if (access) R.string.access_on else R.string.access_off)
        binding.sentinelButton.setText(if (OrionSentinelService.running)
            R.string.sentinel_on else R.string.sentinel_off)
        if (OrionSentinelService.running &&
            (lastSentWatching != watching || lastSentSpeaking != speakingActive)) {
            OrionSentinelService.update(this, watching, speakingActive)
            lastSentWatching = watching
            lastSentSpeaking = speakingActive
        }
    }

    private fun requestPhonePermission(permission: String, callback: (Boolean) -> Unit) {
        if (pendingPhonePermission != null) { callback(false); return }
        pendingPhonePermission = callback
        phonePermission.launch(permission)
    }

    private fun routeNativeAction(action: JSONObject) {
        intentRouter.route(action,
            authorize = { details, execute ->
                requestAuthorization(details, execute, deny = {})
            },
            requestPermission = ::requestPhonePermission,
            report = { outcome ->
                val detail = outcome.optString("detail", outcome.optString("error", "Phone action failed"))
                if (outcome.optBoolean("ok") && outcome.has("numbers")) {
                    val numbers = outcome.optJSONArray("numbers")
                    val emails = outcome.optJSONArray("emails")
                    val text = buildString {
                        append("Numbers:\n")
                        if (numbers != null) for (index in 0 until numbers.length()) append(numbers.optString(index)).append('\n')
                        append("\nE-mail addresses:\n")
                        if (emails != null) for (index in 0 until emails.length()) append(emails.optString(index)).append('\n')
                    }
                    MaterialAlertDialogBuilder(this).setTitle(outcome.optString("name", "Phone contact"))
                        .setMessage(text).setPositiveButton("Close", null).show()
                } else Toast.makeText(this, detail, Toast.LENGTH_LONG).show()
            },
        )
    }

    private fun routeDeviceCommand(request: JSONObject) {
        val id = request.optString("id")
        val method = request.optString("method")
        val params = request.optJSONObject("params") ?: JSONObject()
        if (id.isBlank() || !request.optBoolean("requires_local_approval", false)) return
        if (!OrionFrameTransport.isCommandPending(id)) return
        fun reject(message: String) {
            OrionFrameTransport.sendCommandError(id, -32001, message)
        }
        fun stillPending(): Boolean {
            if (OrionFrameTransport.isCommandPending(id)) return true
            reject("Command expired or disconnected")
            return false
        }
        fun completed(result: JSONObject) {
            if (!result.optBoolean("ok")) {
                reject(result.optString("error", result.optString("detail", "Action failed")))
            } else {
                result.put("authorized", true)
                OrionFrameTransport.sendCommandResult(id, result)
            }
        }
        when (method) {
            "viewport.capture_dom" -> requestAuthorization(
                "TARGET: ACTIVE SCREEN\nACTION: Read visible UI labels and controls",
                approve = {
                    if (stillPending()) OrionAccessibilityService.requestSnapshot { snapshot ->
                        if (!stillPending()) return@requestSnapshot
                        if (snapshot.optBoolean("ok")) {
                            approvedSnapshotId = snapshot.optString("snapshotId")
                            approvedSnapshot = snapshot
                            completed(JSONObject().put("ok", true).put("snapshot", snapshot))
                        } else completed(snapshot)
                    }
                }, deny = { reject("Screen read denied") }, commandId = id,
            )
            "viewport.click", "viewport.scroll", "viewport.set_text" -> {
                val snapshotId = params.optString("snapshotId")
                val nodeId = params.optString("nodeId")
                if (snapshotId.isBlank() || snapshotId != approvedSnapshotId) {
                    reject("Approve a fresh screen snapshot first")
                    return
                }
                if (method == "viewport.set_text" && (params.opt("text") !is String || params.optString("text").length > 4000)) {
                    reject("Specify reviewed text of at most 4,000 characters"); return
                }
                val actionName = when (method) {
                    "viewport.click" -> "Click"
                    "viewport.set_text" -> "Enter reviewed text:\n${params.optString("text")}"
                    else -> "Scroll ${params.optString("direction")}"
                }
                val target = snapshotNodeLabel(approvedSnapshot, nodeId)
                requestAuthorization(
                    "TARGET: ${approvedSnapshot?.optString("package").orEmpty()} / $nodeId\n" +
                        "LABEL: $target\nACTION: $actionName",
                    approve = {
                        approvedSnapshotId = null
                        approvedSnapshot = null
                        if (!stillPending()) return@requestAuthorization
                        if (method == "viewport.click") {
                            OrionAccessibilityService.clickNode(snapshotId, nodeId, ::completed)
                        } else if (method == "viewport.set_text") {
                            OrionAccessibilityService.setNodeText(snapshotId, nodeId, params.optString("text"), ::completed)
                        } else {
                            OrionAccessibilityService.scrollNode(
                                snapshotId, nodeId, params.optString("direction"), ::completed)
                        }
                    }, commandId = id,
                    deny = {
                        approvedSnapshotId = null
                        approvedSnapshot = null
                        reject("Gesture denied")
                    },
                )
            }
            in PhoneCommandContract.deviceMethods -> {
                val kind = method.removePrefix("device.")
                val action = JSONObject(params.toString()).put("kind", kind)
                intentRouter.route(action,
                    authorize = { details, execute ->
                        if (stillPending()) requestAuthorization(details,
                            approve = { if (stillPending()) execute() },
                            deny = { reject("Device action denied") },
                            commandId = id)
                    },
                    requestPermission = ::requestPhonePermission,
                    report = ::completed,
                    canExecute = { OrionFrameTransport.isCommandPending(id) },
                )
            }
            else -> reject("Unsupported device command")
        }
    }

    private fun snapshotNodeLabel(snapshot: JSONObject?, nodeId: String): String {
        var node = snapshot?.optJSONObject("root") ?: return "[unavailable]"
        for (part in nodeId.split('/').drop(1)) {
            val children = node.optJSONArray("children") ?: return "[unavailable]"
            val child = (0 until children.length()).mapNotNull { children.optJSONObject(it) }
                .firstOrNull { it.optString("id").substringAfterLast('/') == part }
                ?: return "[unavailable]"
            node = child
        }
        fun visibleLabel(item: JSONObject): String {
            item.optString("text").ifBlank { item.optString("contentDescription") }
                .takeIf { it.isNotBlank() }?.let { return it }
            val children = item.optJSONArray("children")
            if (children != null) for (index in 0 until children.length()) {
                children.optJSONObject(index)?.let { child ->
                    visibleLabel(child).takeIf { it.isNotBlank() }?.let { return it }
                }
            }
            return ""
        }
        return visibleLabel(node).ifBlank { node.optString("class") }.take(90)
    }

    private fun requestAuthorization(
        details: String, approve: () -> Unit, deny: () -> Unit,
        commandId: String? = null,
    ) {
        if (activeAuthorization != null) { deny(); return }
        val request = Authorization(details, approve, deny, commandId)
        activeAuthorization = request
        authorizationExpiry = Runnable {
            if (activeAuthorization === request) {
                finishAuthorization(false)
                Toast.makeText(this, R.string.authorization_expired,
                    Toast.LENGTH_LONG).show()
            }
        }.also { mainHandler.postDelayed(it, AUTHORIZATION_TTL_MS) }
        if (resumed) {
            showForegroundAuthorization(request)
        } else if (Settings.canDrawOverlays(this)) {
            showOverlayAuthorization(request)
        } else {
            finishAuthorization(false)
        }
    }

    private fun showForegroundAuthorization(request: Authorization) {
        binding.authorizationDetails.text = request.details
        binding.authorizationSliderSlot.removeAllViews()
        binding.authorizationSliderSlot.addView(SlideToConfirmView(this).apply {
            onConfirmed = { finishAuthorization(true) }
        })
        binding.authorizationDeck.visibility = View.VISIBLE
    }

    private fun removeAuthorizationOverlay() {
        authorizationOverlay?.let {
            try { (getSystemService(WINDOW_SERVICE) as WindowManager).removeView(it) }
            catch (_: Exception) {}
        }
        authorizationOverlay = null
    }

    private fun showOverlayAuthorization(request: Authorization) {
        val density = resources.displayMetrics.density
        val pad = (16 * density).toInt()
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            setBackgroundColor(Color.rgb(18, 18, 22))
            addView(TextView(this@MainActivity).apply {
                text = getString(R.string.authorization_required)
                setTextColor(Color.rgb(211, 47, 47))
                textSize = 12f
                typeface = Typeface.MONOSPACE
            })
            addView(ScrollView(this@MainActivity).apply { addView(TextView(this@MainActivity).apply {
                text = request.details
                setTextColor(Color.rgb(242, 230, 232))
                textSize = 12f
                typeface = Typeface.MONOSPACE
                setPadding(0, pad / 2, 0, pad / 2)
            }) }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (180 * density).toInt()))
            addView(SlideToConfirmView(this@MainActivity).apply {
                onConfirmed = { finishAuthorization(true) }
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, (56 * density).toInt()))
            addView(Button(this@MainActivity).apply {
                text = getString(R.string.deny)
                setOnClickListener { finishAuthorization(false) }
            })
        }
        val lp = WindowManager.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
            android.graphics.PixelFormat.TRANSLUCENT,
        ).apply { gravity = Gravity.BOTTOM }
        try {
            (getSystemService(WINDOW_SERVICE) as WindowManager).addView(panel, lp)
            authorizationOverlay = panel
        } catch (_: Exception) {
            finishAuthorization(false)
        }
    }

    private fun finishAuthorization(approved: Boolean) {
        val request = activeAuthorization ?: return
        activeAuthorization = null
        authorizationExpiry?.let(mainHandler::removeCallbacks)
        authorizationExpiry = null
        if (::binding.isInitialized) binding.authorizationDeck.visibility = View.GONE
        removeAuthorizationOverlay()
        if (approved) mainHandler.postDelayed(request.approve, 250)
        else request.deny()
    }

    // ── WebView ───────────────────────────────────────────────────────────────

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        val web = binding.webView
        web.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            // ORION talks back the instant a reply lands — no user gesture gate.
            mediaPlaybackRequiresUserGesture = false
            cacheMode = WebSettings.LOAD_DEFAULT
            useWideViewPort = true
            loadWithOverviewMode = true
        }

        // Native phone actions for ORION's web UI (window.OrionNative.*): call,
        // text, e-mail, navigate, share — the "more advanced than the HTML" part.
        web.addJavascriptInterface(NativeBridge(
            this,
            isTrustedPage = { trustedPageActive },
            onEndpointsSaved = { onEndpointsSaved() },
            onAction = { action -> routeNativeAction(action) },
            onTelemetry = { model, microphone ->
                if (model.isNotBlank()) modelRoute = model
                if (microphone != null) micActive = microphone
                updateTelemetry()
            },
            onSpeaking = { active ->
                speakingActive = active
                updateTelemetry()
            },
            onSpeak = { text ->
                if (isTrustedOrigin(binding.webView.url)) phoneSpeech?.speak(text)
            },
            onStopSpeaking = { RemoteSpeech.stopAll() },
            onPrepareDictation = {
                if (isTrustedOrigin(binding.webView.url)) {
                    OrionAudioService.stop(this)
                    phoneSpeech?.stop()
                    updateTelemetry()
                }
            },
            onStartDictation = { startPhoneDictation() },
            onFinishDictation = { phoneDictation?.finish() },
        ), "OrionNative")

        web.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                runOnUiThread {
                    // Only ORION's own page may open the microphone (getUserMedia).
                    val wantsMic = PermissionRequest.RESOURCE_AUDIO_CAPTURE in request.resources
                    if (!wantsMic || !isTrustedOrigin(request.origin?.toString())) {
                        request.deny()
                        return@runOnUiThread
                    }
                    if (ContextCompat.checkSelfPermission(
                            this@MainActivity, Manifest.permission.RECORD_AUDIO
                        ) == PackageManager.PERMISSION_GRANTED
                    ) {
                        request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
                    } else {
                        pendingMicRequest?.deny()
                        pendingMicRequest = request
                        micPermission.launch(Manifest.permission.RECORD_AUDIO)
                    }
                }
            }

            override fun onPermissionRequestCanceled(request: PermissionRequest) {
                if (pendingMicRequest === request) pendingMicRequest = null
            }

            /** The page's pairing-code prompt(), as a native dialog instead of
             *  WebView's bare "The page at https://… says:" box. */
            override fun onJsPrompt(
                view: WebView, url: String, message: String?,
                defaultValue: String?, result: JsPromptResult,
            ): Boolean {
                if (!isTrustedOrigin(url) || isFinishing) return false
                showPairingDialog(message, defaultValue, result)
                return true
            }
        }

        web.webViewClient = object : WebViewClient() {
            private fun failMainFrame(url: String, detail: String) {
                if (loadFailed || originOf(url) !=
                    originOf(candidates.getOrNull(candidateIndex).orEmpty())) return
                trustedPageActive = false
                loadFailed = true
                if (candidateIndex < candidates.size - 1) {
                    candidateIndex++
                    binding.connectingView.visibility = View.VISIBLE
                    loadCurrent()
                } else showError(detail)
                binding.swipeRefresh.isRefreshing = false
            }

            override fun onPageStarted(view: WebView?, url: String?, favicon: Bitmap?) {
                trustedPageActive = false
            }

            override fun onPageCommitVisible(view: WebView?, url: String?) {
                if (!loadFailed && isTrustedOrigin(url) &&
                    originOf(url.orEmpty()) ==
                    originOf(candidates.getOrNull(candidateIndex).orEmpty())) {
                    trustedPageActive = true
                }
            }

            override fun shouldOverrideUrlLoading(
                view: WebView, request: WebResourceRequest
            ): Boolean {
                val uri = request.url
                // The JavaScript bridge is injected into frames as well as the
                // main page. Never let a third-party frame load inside it or
                // turn a frame navigation into a phone action.
                if (!request.isForMainFrame) return !isTrustedOrigin(uri.toString())
                return when (uri.scheme?.lowercase()) {
                    null -> false
                    "http", "https" -> {
                        // ONLY ORION's own uplink (any of his known hosts) stays
                        // in the WebView — it is the sole origin trusted with the
                        // native bridge. Any other site (a news article he opens)
                        // is handed to the system browser, so no third-party page
                        // can ever reach window.OrionNative.
                        if (isTrustedOrigin(uri.toString())) {
                            false
                        } else {
                            launchExternal(uri)
                            true
                        }
                    }
                    // tel:, sms:, smsto:, mailto:, geo:, google.navigation:,
                    // whatsapp:, tg:, … → the phone's own app (call/text/navigate).
                    else -> {
                        launchExternal(uri)
                        true
                    }
                }
            }

            override fun onReceivedSslError(
                view: WebView, handler: SslErrorHandler, error: SslError
            ) {
                // Named HTTPS endpoints must pass Android's normal certificate
                // and host checks. A saved fingerprint cannot make an invalid
                // public certificate safe for the WebView to load. The one
                // exception is ORION's home Wi-Fi link: a private LAN address
                // whose certificate is exactly the one pinned at pairing.
                if (isTrustedOrigin(error.url) &&
                    LanTrust.webViewCertificateMatches(this@MainActivity, error.url, error.certificate)) {
                    handler.proceed()
                    return
                }
                handler.cancel()
                if (isTrustedOrigin(error.url))
                    failMainFrame(error.url, getString(R.string.error_https_certificate))
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                binding.swipeRefresh.isRefreshing = false
                // A failed load also "finishes" (with the error page) — only a
                // real page from the endpoint we asked for ends "Connecting…".
                if (!loadFailed && isTrustedOrigin(url) &&
                    originOf(url.orEmpty()) ==
                    originOf(candidates.getOrNull(candidateIndex).orEmpty())) {
                    trustedPageActive = true
                    pageLoadTimeout?.let(mainHandler::removeCallbacks)
                    pageLoadTimeout = null
                    retryAttempt = 0
                    retryConnection?.let(mainHandler::removeCallbacks)
                    retryConnection = null
                    hideConnecting()
                    lastAudioStatus = ""
                    captureCredentials()
                }
            }

            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?, error: WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    failMainFrame(request.url.toString(), describeFailure(
                        request.url.toString(), error?.description?.toString().orEmpty()))
                }
                binding.swipeRefresh.isRefreshing = false
            }

            override fun onReceivedHttpError(
                view: WebView?, request: WebResourceRequest?, errorResponse: WebResourceResponse?
            ) {
                if (request?.isForMainFrame == true && errorResponse != null &&
                    errorResponse.statusCode >= 400) {
                    failMainFrame(request.url.toString(),
                        "${displayAddress(request.url.toString())} — HTTP ${errorResponse.statusCode}")
                }
            }

            /** The page's renderer died — usually Android reclaiming memory from
             *  the 3-D face while ORION was in the background. Unhandled, that
             *  kills the whole app; instead rebuild the screen. */
            override fun onRenderProcessGone(
                view: WebView, detail: RenderProcessGoneDetail
            ): Boolean {
                trustedPageActive = false
                webViewGone = true
                credentialScript = null
                pendingMicRequest = null
                (view.parent as? ViewGroup)?.removeView(view)
                view.destroy()
                recreate()
                return true
            }
        }
    }

    private fun showPairingDialog(message: String?, defaultValue: String?, result: JsPromptResult) {
        val density = resources.displayMetrics.density
        val input = EditText(this).apply {
            setText(defaultValue.orEmpty())
            hint = getString(R.string.pair_hint)
            isSingleLine = true
            inputType = InputType.TYPE_CLASS_TEXT or
                InputType.TYPE_TEXT_FLAG_CAP_CHARACTERS or
                InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
        }
        val box = FrameLayout(this).apply {
            val side = (24 * density).toInt()
            setPadding(side, (4 * density).toInt(), side, 0)
            addView(input)
        }
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.pair_title)
            .setMessage(if (message.isNullOrBlank()) getString(R.string.pair_message) else message)
            .setView(box)
            .setPositiveButton(R.string.pair_action) { _, _ ->
                result.confirm(input.text.toString().trim())
            }
            .setNegativeButton(R.string.cancel) { _, _ -> result.cancel() }
            .setOnCancelListener { result.cancel() }
            .show()
        input.requestFocus()
    }

    /** Pull-to-refresh only from ORION's header bar. The page never scrolls as a
     *  whole (the chat log is an inner scroller), so SwipeRefreshLayout took
     *  every downward drag for a refresh — scrolling back through the
     *  conversation reloaded the page and wiped it. */
    private fun configurePullToRefresh() {
        binding.swipeRefresh.setColorSchemeResources(R.color.orion_accent)
        binding.swipeRefresh.setProgressBackgroundColorSchemeResource(R.color.orion_surface)
        binding.swipeRefresh.setOnChildScrollUpCallback { parent, _ ->
            val origin = IntArray(2)
            parent.getLocationOnScreen(origin)
            touchDownRawY - origin[1] > PULL_ZONE_DP * resources.displayMetrics.density
        }
        binding.swipeRefresh.setOnRefreshListener {
            if (webViewGone) binding.swipeRefresh.isRefreshing = false
            else binding.webView.reload()
        }
    }

    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN) touchDownRawY = ev.rawY
        return super.dispatchTouchEvent(ev)
    }

    // ── window ────────────────────────────────────────────────────────────────

    private fun applyImmersiveKeepAwake() {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).apply {
            hide(WindowInsetsCompat.Type.systemBars())
            systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    /** Edge-to-edge means the system no longer shrinks the window for the
     *  keyboard, so ORION's message box (at the bottom of the page) sat hidden
     *  under it while typing. Pad the screen by the keyboard and any notch. */
    private fun applyInsets() {
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            val cutout = insets.getInsets(WindowInsetsCompat.Type.displayCutout())
            view.setPadding(cutout.left, cutout.top, cutout.right, maxOf(ime.bottom, cutout.bottom))
            insets
        }
    }

    private fun showConnecting(detail: String) {
        binding.connectingDetail.text = detail
        binding.connectingView.visibility = View.VISIBLE
    }

    private fun hideConnecting() {
        binding.connectingView.visibility = View.GONE
    }

    private fun describeFailure(url: String, description: String): String {
        val reason = description.removePrefix("net::").ifEmpty { "no response" }
        val hint = when {
            url.startsWith("https:", ignoreCase = true) && description.contains("SSL", ignoreCase = true) ->
                getString(R.string.hint_not_https)
            url.startsWith("http:", ignoreCase = true) &&
                (description.contains("EMPTY_RESPONSE") || description.contains("CONNECTION_CLOSED")) ->
                getString(R.string.hint_is_https)
            else -> ""
        }
        return displayAddress(url) + " — " + reason + if (hint.isEmpty()) "" else "\n\n$hint"
    }

    private fun showError(detail: String) {
        trustedPageActive = false
        connectAttempt++                          // drop any probe still in flight
        pageLoadTimeout?.let(mainHandler::removeCallbacks)
        pageLoadTimeout = null
        hideConnecting()
        binding.errorDetail.text = detail
        binding.errorDetail.visibility = if (detail.isEmpty()) View.GONE else View.VISIBLE
        binding.errorView.visibility = View.VISIBLE
        binding.swipeRefresh.visibility = View.GONE
        retryConnection?.let(mainHandler::removeCallbacks)
        val ceiling = min(30_000L, 1_000L shl min(retryAttempt++, 5))
        val delay = Random.nextLong(ceiling / 2, ceiling + 1)
        retryConnection = Runnable {
            if (!isFinishing && !isDestroyed) connect()
        }.also { mainHandler.postDelayed(it, delay) }
    }

    private fun hideError() {
        binding.errorView.visibility = View.GONE
        binding.swipeRefresh.visibility = View.VISIBLE
    }

    override fun onDestroy() {
        pcControls?.close()
        trustedPageActive = false
        phoneDictation?.close()
        phoneDictation = null
        phoneSpeech?.close()
        phoneSpeech = null
        if (networkCallbackRegistered) {
            try {(getSystemService(CONNECTIVITY_SERVICE) as ConnectivityManager)
                .unregisterNetworkCallback(networkCallback)} catch (_: Exception) { }
        }
        retryConnection?.let(mainHandler::removeCallbacks)
        pageLoadTimeout?.let(mainHandler::removeCallbacks)
        networkReconnect?.let(mainHandler::removeCallbacks)
        mainHandler.removeCallbacks(telemetryTicker)
        authorizationExpiry?.let(mainHandler::removeCallbacks)
        authorizationExpiry = null
        activeAuthorization?.deny?.invoke()
        activeAuthorization = null
        removeAuthorizationOverlay()
        OrionFrameTransport.setListener(null)
        OrionFrameTransport.stop()
        if (OrionProjectionService.isCapturing) OrionProjectionService.stop(this)
        try {
            pendingMicRequest?.deny()
        } catch (_: Exception) {
        }
        pendingMicRequest = null
        if (::binding.isInitialized && !webViewGone) {
            try {
                val web = binding.webView
                (web.parent as? ViewGroup)?.removeView(web)
                web.stopLoading()
                web.destroy()
            } catch (_: Exception) {
            }
        }
        super.onDestroy()
    }

    companion object {
        private const val AUTHORIZATION_TTL_MS = 80_000L
        const val EXTRA_RECONFIGURED = "com.orion.companion.RECONFIGURED"
        private const val PROBE_TIMEOUT_MS = 4000
        private const val PULL_ZONE_DP = 64
        private const val READ_CREDENTIALS_JS =
            "(function(){try{return JSON.stringify({d:localStorage.getItem('orion_device')||''," +
                "r:localStorage.getItem('orion_refresh')||''});}catch(e){return '';}})()"
    }
}
