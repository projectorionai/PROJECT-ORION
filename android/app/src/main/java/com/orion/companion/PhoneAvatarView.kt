package com.orion.companion

import android.content.Context
import android.graphics.Color
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import android.webkit.RenderProcessGoneDetail
import android.webkit.WebSettings
import android.widget.FrameLayout
import android.widget.TextView
import androidx.webkit.WebViewAssetLoader
import org.json.JSONObject
import java.io.ByteArrayInputStream

/**
 * ORION's face on the phone: the PC's own Screen Face (Sentinel design by
 * default), drawn by assets/orion_face.html - a Canvas 2-D port generated from
 * the desktop source by tools/export_phone_avatar.py. It is bundled, so it
 * works with the PC and the network off. No Three.js, no orb.
 *
 * Driven two ways: the phone's own listening/thinking/speaking state and the
 * feelings in what is said (the desktop's sentiment lexicon, inside the page),
 * or - while [link] is on - the PC's state and emotions from its uplink.
 */
class PhoneAvatarView(context: Context) : FrameLayout(context) {
    private companion object {
        const val HOST = "appassets.androidplatform.net"
        const val PAGE = "https://$HOST/assets/orion_face.html"
        const val AMPLITUDE_MS = 66L
    }

    private val assets = WebViewAssetLoader.Builder()
        .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(context.applicationContext))
        .build()
    private val fallback = TextView(context).apply {
        text = "Face unavailable"
        textSize = 12f
        setTextColor(Color.rgb(159, 152, 156))
        gravity = Gravity.CENTER
    }
    private var browser: WebView? = null
    private var ready = false
    private var disposed = false
    private var resumed = true
    private var bootStartedAt = 0L
    private var lastSentAt = 0L
    private var wantedState = "STANDBY"
    private var sentState = ""
    private var wantedSpeaking = false
    private var sentSpeaking: Boolean? = null
    private var wantedAmplitude = 0f
    private var linked = false
    private val pending = ArrayList<String>()

    /** A live loudness (speaker or microphone), polled ~15 times a second
     *  while he listens or speaks, so the mouth and the level ticks move
     *  with the sound rather than with the console's slower status tick. */
    var amplitudeSource: (() -> Float)? = null

    private val amplitudeTick = object : Runnable {
        override fun run() {
            if (disposed || !resumed || !ready) return
            val source = amplitudeSource
            if (source != null && (wantedSpeaking || wantedState == "LISTENING")) {
                val level = runCatching(source).getOrDefault(0f).coerceIn(0f, 1f)
                browser?.evaluateJavascript("window.orionAmplitude($level);", null)
            }
            postDelayed(this, AMPLITUDE_MS)
        }
    }

    private val bootCheck = object : Runnable {
        override fun run() {
            val view = browser ?: return
            if (disposed || !resumed) return
            view.evaluateJavascript("typeof window.orionStep === 'function' && !window.__orionFailed") { result ->
                if (disposed || browser !== view) return@evaluateJavascript
                if (result == "true") {
                    ready = true
                    fallback.visibility = View.GONE
                    applyStyle(FaceStyle.load(context))
                    pending.forEach { view.evaluateJavascript(it, null) }
                    pending.clear()
                    sendState(force = true)
                    removeCallbacks(amplitudeTick)
                    post(amplitudeTick)
                } else if (SystemClock.elapsedRealtime() - bootStartedAt >= 12_000L) {
                    showFallback()
                } else postDelayed(this, 700L)
            }
        }
    }

    init {
        setBackgroundColor(Color.rgb(6, 6, 6))
        addView(fallback, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        try {
            val view = WebView(context)
            view.setBackgroundColor(Color.rgb(6, 6, 6))
            view.settings.javaScriptEnabled = true
            view.settings.domStorageEnabled = false
            view.settings.allowFileAccess = false
            view.settings.allowContentAccess = false
            view.settings.mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            view.settings.setSupportZoom(false)
            view.webViewClient = object : WebViewClient() {
                override fun shouldInterceptRequest(view: WebView?, request: WebResourceRequest): WebResourceResponse? {
                    val url = request.url
                    if (url.scheme == "https" && url.host == HOST && url.path == "/assets/orion_face.html")
                        assets.shouldInterceptRequest(url)?.let { return it }
                    // The face fetches nothing else: refuse every other load.
                    return WebResourceResponse("text/plain", "UTF-8", ByteArrayInputStream(ByteArray(0)))
                }

                override fun shouldOverrideUrlLoading(view: WebView?, request: WebResourceRequest): Boolean = true

                override fun onPageFinished(view: WebView?, url: String?) {
                    if (!disposed && view === browser) {
                        removeCallbacks(bootCheck)
                        post(bootCheck)
                    }
                }

                override fun onReceivedError(view: WebView?, request: WebResourceRequest?,
                                             error: android.webkit.WebResourceError?) {
                    if (request?.isForMainFrame == true) post { showFallback() }
                }

                override fun onRenderProcessGone(view: WebView?, detail: RenderProcessGoneDetail?): Boolean {
                    // Android requires a dead WebView to leave the hierarchy at once.
                    showFallback()
                    return true
                }
            }
            browser = view
            addView(view, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
            bootStartedAt = SystemClock.elapsedRealtime()
            view.loadUrl(PAGE)
            postDelayed(bootCheck, 12_000L)
        } catch (_: RuntimeException) {
            showFallback()
        }
    }

    /** The phone's own activity (or the PC's state while linked). */
    fun show(state: String, speaking: Boolean, amplitude: Float) {
        wantedState = PhoneAvatarPolicy.faceState(state)
        wantedSpeaking = speaking
        wantedAmplitude = amplitude.coerceIn(0f, 1f)
        sendState(force = false)
    }

    /** An emotion by name (the PC's EmotionStateManager output). */
    fun emotion(name: String, intensity: Double = 1.0) {
        if (name.isBlank() || name.length > 40) return
        val level = intensity.coerceIn(0.0, 1.0)
        run("window.orionEmotion(${JSONObject.quote(name)},{intensity:$level});")
    }

    /** Words said on the phone: the page reads their feeling with the PC's
     *  lexicon and wears the PC's response to it (ignored while linked). */
    fun feel(text: String, fromUser: Boolean) {
        if (text.isBlank()) return
        val origin = if (fromUser) "user" else "orion"
        run("window.orionFeel(${JSONObject.quote(text.take(4000))},'$origin');")
    }

    /** True while the face follows the PC uplink instead of the phone. */
    fun link(on: Boolean) {
        if (on == linked) return
        linked = on
        run("window.orionLink(${if (on) "true" else "false"});")
    }

    /** The PC's design and colours; remembered so the face matches offline. */
    fun style(style: FaceStyle) {
        FaceStyle.save(context, style)
        applyStyle(style)
    }

    private fun applyStyle(style: FaceStyle) {
        run("window.orionDesign(${JSONObject.quote(style.design)});" +
            "window.orionPalette(${JSONObject.quote(style.primary)},${JSONObject.quote(style.accent)}," +
            "${JSONObject.quote(style.background)});")
    }

    private fun run(script: String) {
        val view = browser ?: return
        if (disposed) return
        if (!ready) {
            if (pending.size < 32) pending.add(script)
            return
        }
        view.evaluateJavascript(script, null)
    }

    private fun sendState(force: Boolean) {
        val view = browser ?: return
        if (!ready || !resumed || disposed) return
        val now = SystemClock.elapsedRealtime()
        val changed = wantedState != sentState || wantedSpeaking != sentSpeaking
        if (!force && !changed && now - lastSentAt < 120L) return
        lastSentAt = now
        sentState = wantedState
        sentSpeaking = wantedSpeaking
        val script = "window.orionState('$wantedState');window.orionSpeaking($wantedSpeaking);" +
            (if (amplitudeSource == null || !(wantedSpeaking || wantedState == "LISTENING"))
                "window.orionAmplitude($wantedAmplitude);" else "")
        view.evaluateJavascript(script, null)
    }

    fun resumeVisual() {
        resumed = true
        browser?.onResume()
        if (ready) {
            sendState(force = true)
            removeCallbacks(amplitudeTick)
            post(amplitudeTick)
        } else post(bootCheck)
    }

    fun pauseVisual() {
        resumed = false
        removeCallbacks(bootCheck)
        removeCallbacks(amplitudeTick)
        browser?.onPause()
    }

    private fun showFallback() {
        if (disposed) return
        ready = false
        removeCallbacks(bootCheck)
        removeCallbacks(amplitudeTick)
        val view = browser
        browser = null
        if (view != null) {
            removeView(view)
            runCatching { view.stopLoading(); view.destroy() }
        }
        fallback.visibility = View.VISIBLE
    }

    fun dispose() {
        pauseVisual()
        showFallback()
        disposed = true
    }
}

/** The desktop Screen Face's design and colours, as the PC last sent them. */
data class FaceStyle(val design: String, val primary: String, val accent: String, val background: String) {
    companion object {
        private const val PREFS = "orion_face"
        val DEFAULT = FaceStyle("sentinel", "#ff2038", "#ededed", "#060606")
        private val COLOUR = Regex("#[0-9a-fA-F]{6}")
        private val DESIGNS = setOf("sentinel", "arc", "visor", "minimal", "companion")

        /** Only a known design and plain #rrggbb colours are accepted. */
        fun parse(json: JSONObject?): FaceStyle? {
            if (json == null) return null
            val design = json.optString("design").lowercase().takeIf { it in DESIGNS } ?: return null
            fun colour(key: String, default: String) =
                json.optString(key).takeIf { COLOUR.matches(it) }?.lowercase() ?: default
            return FaceStyle(design, colour("primary", DEFAULT.primary),
                colour("accent", DEFAULT.accent), colour("background", DEFAULT.background))
        }

        fun load(context: Context): FaceStyle {
            val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            return parse(JSONObject().put("design", p.getString("design", DEFAULT.design))
                .put("primary", p.getString("primary", DEFAULT.primary))
                .put("accent", p.getString("accent", DEFAULT.accent))
                .put("background", p.getString("background", DEFAULT.background))) ?: DEFAULT
        }

        fun save(context: Context, style: FaceStyle) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString("design", style.design).putString("primary", style.primary)
                .putString("accent", style.accent).putString("background", style.background).apply()
        }
    }
}
