package com.orion.companion

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.Uri
import android.net.http.SslCertificate
import android.net.http.SslError
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.MotionEvent
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
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.Toast
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
import java.net.InetSocketAddress
import java.net.Socket
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicIntegerArray

/**
 * The native shell: a full-screen WebView onto ORION's uplink. His voxel
 * orb/face, live emotion mirror, spoken replies and memory all come from the
 * uplink the desktop already serves — this app makes it a first-class Android
 * app: native config, microphone permission, self-signed-TLS handling, keep-
 * awake and immersive full-screen.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    /** The endpoints that answered the last reachability probe, best first —
     *  what the WebView loads and fails over through. */
    private var candidates: List<String> = emptyList()
    private var candidateIndex = 0
    private var trustedHosts: Set<String> = emptySet()

    /** Bumped by every connect(), so a slow probe from an earlier attempt can't
     *  load over a newer one. */
    private var connectAttempt = 0
    private var loadFailed = false
    private var resumedOnce = false
    private var webViewGone = false
    private var touchDownRawY = 0f
    private var credentialScript: ScriptHandler? = null
    private var pendingMicRequest: PermissionRequest? = null

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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Prefs.candidateUrls(this).isEmpty()) {
            startActivity(Intent(this, SetupActivity::class.java))
            finish()
            return
        }
        trustedHosts = Prefs.trustedHosts(this)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        applyImmersiveKeepAwake()
        applyInsets()
        configureWebView()
        configurePullToRefresh()
        installCredentialSeed()

        binding.settingsButton.setOnClickListener { openSetup() }
        binding.changeServerButton.setOnClickListener { openSetup() }
        binding.retryButton.setOnClickListener { connect() }

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
        if (intent.getBooleanExtra(EXTRA_RECONFIGURED, false) && ::binding.isInitialized) {
            trustedHosts = Prefs.trustedHosts(this)
            installCredentialSeed()
            connect()
        }
    }

    override fun onResume() {
        super.onResume()
        if (!::binding.isInitialized || webViewGone) return
        binding.webView.onResume()
        if (resumedOnce) recheckConnection() else resumedOnce = true
    }

    override fun onPause() {
        if (::binding.isInitialized && !webViewGone) {
            captureCredentials()
            binding.webView.onPause()
        }
        super.onPause()
    }

    // ── connecting ────────────────────────────────────────────────────────────

    /** Probe every known endpoint at once, then load the best one that answers. */
    private fun connect() {
        val attempt = ++connectAttempt
        val all = Prefs.candidateUrls(this)
        hideError()
        showConnecting(all.firstOrNull()?.let(::displayAddress).orEmpty())
        Thread {
            val reachable = probe(all)
            runOnUiThread {
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
     * in priority order (Tailscale name → Tailscale IP → LAN). Without this an
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
        host.isNotEmpty() && Socket().use {
            it.connect(InetSocketAddress(host, port), PROBE_TIMEOUT_MS)
            true
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
        loadFailed = false
        binding.connectingDetail.text = displayAddress(url)
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

    private fun isTrustedHost(host: String?): Boolean {
        val h = host.orEmpty()
        return h.isNotEmpty() && trustedHosts.any { it.equals(h, ignoreCase = true) }
    }

    private fun hostOf(url: String): String = try {
        Uri.parse(url).host.orEmpty()
    } catch (_: Exception) {
        ""
    }

    private fun displayAddress(url: String): String = url.substringBefore('?').trimEnd('/')

    private fun openSetup() {
        // Setup sits on top of this screen, so Back returns here unchanged; a
        // new address comes back through onNewIntent().
        startActivity(Intent(this, SetupActivity::class.java))
    }

    /** Hand a non-web URI (tel:, sms:, mailto:, geo:, …) to the phone's app. */
    private fun launchExternal(uri: Uri) {
        try {
            startActivity(Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        } catch (_: Exception) {
            Toast.makeText(this, "No app to handle ${uri.scheme}: links.", Toast.LENGTH_LONG).show()
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
        val host = uri.host.orEmpty()
        if (scheme == null || host.isEmpty()) null
        else "$scheme://$host" + (if (uri.port > 0) ":${uri.port}" else "")
    } catch (_: Exception) {
        null
    }

    /** Read the pairing the page holds for the current origin and keep a copy. */
    private fun captureCredentials() {
        if (webViewGone) return
        val web = binding.webView
        if (!isTrustedHost(hostOf(web.url.orEmpty()))) return
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
                }
            }
        } catch (_: Exception) {
        }
    }

    /** The page just paired or refreshed its token: learn the new endpoints and
     *  keep a copy of the pairing for the other endpoints. */
    private fun onEndpointsSaved() {
        if (isFinishing || isDestroyed) return
        trustedHosts = Prefs.trustedHosts(this)
        installCredentialSeed()
        captureCredentials()
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
        web.addJavascriptInterface(NativeBridge(this) { onEndpointsSaved() }, "OrionNative")

        web.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                runOnUiThread {
                    // Only ORION's own page may open the microphone (getUserMedia).
                    val wantsMic = PermissionRequest.RESOURCE_AUDIO_CAPTURE in request.resources
                    if (!wantsMic || !isTrustedHost(request.origin?.host)) {
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
                if (!isTrustedHost(hostOf(url)) || isFinishing) return false
                showPairingDialog(message, defaultValue, result)
                return true
            }
        }

        web.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView, request: WebResourceRequest
            ): Boolean {
                val uri = request.url
                return when (uri.scheme?.lowercase()) {
                    null -> false
                    "http", "https" -> {
                        // ONLY ORION's own uplink (any of his known hosts) stays
                        // in the WebView — it is the sole origin trusted with the
                        // native bridge. Any other site (a news article he opens)
                        // is handed to the system browser, so no third-party page
                        // can ever reach window.OrionNative.
                        if (isTrustedHost(uri.host)) {
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
                // ORION serves a self-signed certificate. Proceed only for one
                // of his known hosts (LAN or Tailscale), and only for the
                // certificate first accepted from them: the host name alone is
                // something anyone on the same network can answer for.
                if (!isTrustedHost(hostOf(error.url ?: ""))) {
                    handler.cancel()
                    return
                }
                val presented = sha256Of(error.certificate)
                val pinned = Prefs.certPin(this@MainActivity)
                when {
                    presented == null -> handler.cancel()
                    pinned.isEmpty() -> {
                        Prefs.setCertPin(this@MainActivity, presented)
                        handler.proceed()
                    }
                    pinned.equals(presented, ignoreCase = true) -> handler.proceed()
                    else -> {
                        handler.cancel()
                        showError(getString(R.string.error_cert_changed))
                    }
                }
            }

            override fun onPageFinished(view: WebView?, url: String?) {
                binding.swipeRefresh.isRefreshing = false
                // A failed load also "finishes" (with the error page) — only a
                // real page from the endpoint we asked for ends "Connecting…".
                val expected = candidates.getOrNull(candidateIndex)?.let(::hostOf).orEmpty()
                if (!loadFailed && hostOf(url.orEmpty()).equals(expected, ignoreCase = true)) {
                    hideConnecting()
                }
            }

            override fun onReceivedError(
                view: WebView?, request: WebResourceRequest?, error: WebResourceError?
            ) {
                if (request?.isForMainFrame == true) {
                    loadFailed = true
                    // Failover: try the next endpoint that answered (LAN ⇄
                    // Tailscale) before surfacing an error — this is what makes
                    // the app "just work" at home and away without reconfiguring.
                    if (candidateIndex < candidates.size - 1) {
                        candidateIndex++
                        binding.connectingView.visibility = View.VISIBLE
                        loadCurrent()
                    } else {
                        showError(describeFailure(
                            request.url.toString(), error?.description?.toString().orEmpty()))
                    }
                }
                binding.swipeRefresh.isRefreshing = false
            }

            /** The page's renderer died — usually Android reclaiming memory from
             *  the 3-D face while ORION was in the background. Unhandled, that
             *  kills the whole app; instead rebuild the screen. */
            override fun onRenderProcessGone(
                view: WebView, detail: RenderProcessGoneDetail
            ): Boolean {
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

    /** SHA-256 of a certificate's DER encoding, as lowercase hex. */
    private fun sha256Of(certificate: SslCertificate?): String? {
        if (certificate == null) return null
        val der = try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                certificate.x509Certificate?.encoded
            } else {
                // Before API 29 the encoded certificate is only reachable
                // through the state bundle SslCertificate saves itself to.
                SslCertificate.saveState(certificate).getByteArray("x509-certificate")
            }
        } catch (_: Exception) {
            null
        } ?: return null
        return MessageDigest.getInstance("SHA-256").digest(der)
            .joinToString("") { "%02x".format(it) }
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
        connectAttempt++                          // drop any probe still in flight
        hideConnecting()
        binding.errorDetail.text = detail
        binding.errorDetail.visibility = if (detail.isEmpty()) View.GONE else View.VISIBLE
        binding.errorView.visibility = View.VISIBLE
        binding.swipeRefresh.visibility = View.GONE
        binding.settingsButton.visibility = View.GONE
    }

    private fun hideError() {
        binding.errorView.visibility = View.GONE
        binding.swipeRefresh.visibility = View.VISIBLE
        binding.settingsButton.visibility = View.VISIBLE
    }

    override fun onDestroy() {
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
        const val EXTRA_RECONFIGURED = "com.orion.companion.RECONFIGURED"
        private const val PROBE_TIMEOUT_MS = 2500
        private const val PULL_ZONE_DP = 64
        private const val READ_CREDENTIALS_JS =
            "(function(){try{return JSON.stringify({d:localStorage.getItem('orion_device')||''," +
                "r:localStorage.getItem('orion_refresh')||''});}catch(e){return '';}})()"
    }
}
