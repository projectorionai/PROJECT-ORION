package com.orion.companion

import android.app.Activity
import android.webkit.JavascriptInterface
import org.json.JSONObject

/**
 * Native phone actions exposed to ORION's web UI as `window.OrionNative`.
 *
 * This is what makes the app "more advanced than the HTML": from the same
 * conversation, ORION can place a call, start a text, draft an e-mail, open
 * turn-by-turn navigation, or share — using the phone's own apps, from
 * anywhere he's reachable.
 *
 * Calls and texts are proposed to the native authorisation deck. A resolved
 * application or offline map may open directly in local phone mode.
 *
 * The page feature-detects this object: `if (window.OrionNative) { … }`.
 */
class NativeBridge(
    private val activity: Activity,
    /** Must be safe to call from a JavaScript bridge thread. The page is
     *  checked again on the UI thread before any native side effect. */
    private val isTrustedPage: () -> Boolean = { false },
    /** Runs on the UI thread after the page reports its endpoints — which it
     *  does straight after pairing and on every token refresh. */
    private val onEndpointsSaved: () -> Unit = {},
    private val onAction: (JSONObject) -> Unit = {},
    private val onTelemetry: (String, Boolean?) -> Unit = { _, _ -> },
    private val onSpeaking: (Boolean) -> Unit = {},
    private val onSpeak: (String) -> Unit = {},
    private val onStopSpeaking: () -> Unit = {},
    private val onPrepareDictation: () -> Unit = {},
    private val onStartDictation: () -> Unit = {},
    private val onFinishDictation: () -> Unit = {},
) {

    private fun trusted(): Boolean = runCatching { isTrustedPage() }.getOrDefault(false)

    private fun onTrustedPage(action: () -> Unit) {
        if (!trusted()) return
        activity.runOnUiThread { if (trusted()) action() }
    }

    private fun propose(action: JSONObject) = onTrustedPage { onAction(action) }

    /** Propose a call to the native authorisation deck. */
    @JavascriptInterface
    fun call(number: String) {
        propose(JSONObject().put("kind", "make_call").put("number", number))
    }

    /** Propose an SMS to the native authorisation deck. */
    @JavascriptInterface
    fun sms(number: String, body: String) {
        propose(JSONObject().put("kind", "send_sms")
            .put("number", number).put("body", body))
    }

    @JavascriptInterface
    fun openApp(packageName: String) {
        propose(JSONObject().put("kind", "open_app").put("package", packageName))
    }

    /** Open the e-mail composer to [to] with an optional [subject]/[body]. */
    @JavascriptInterface
    fun email(to: String, subject: String, body: String) {
        propose(JSONObject().put("kind", "compose_email").put("to", to)
            .put("subject", subject).put("body", body))
    }

    @JavascriptInterface
    fun sendEmail(to: String, subject: String, body: String) =
        propose(JSONObject().put("kind", "send_email").put("to", to)
            .put("subject", subject).put("body", body))

    /** Open turn-by-turn navigation for a free-text [query] (address / place). */
    @JavascriptInterface
    fun navigate(query: String) {
        propose(JSONObject().put("kind", "navigate").put("query", query))
    }

    /** Show a place / map for a free-text [query] without starting navigation. */
    @JavascriptInterface
    fun map(query: String) {
        propose(JSONObject().put("kind", "map").put("query", query))
    }

    /** Open an external URL in the phone's browser / relevant app. */
    @JavascriptInterface
    fun openUrl(url: String) {
        propose(JSONObject().put("kind", "open_url").put("url", url))
    }

    /** Offer [text] to the Android share sheet. */
    @JavascriptInterface
    fun share(text: String) {
        propose(JSONObject().put("kind", "share").put("text", text))
    }

    /** New phone controls use the same reviewed native boundary as command RPC. */
    @JavascriptInterface
    fun action(json: String) {
        if (json.length > 32_000) return
        runCatching { JSONObject(json) }.getOrNull()?.let(::propose)
    }

    /** Signal a pairing change; endpoint data from JavaScript is never trusted. */
    @JavascriptInterface
    fun endpointsChanged() {
        onTrustedPage { onEndpointsSaved() }
    }

    @JavascriptInterface
    fun deviceStatus(): String = if (!trusted()) "{}" else JSONObject()
        .put("microphone", androidx.core.content.ContextCompat.checkSelfPermission(activity,
            android.Manifest.permission.RECORD_AUDIO) == android.content.pm.PackageManager.PERMISSION_GRANTED)
        .put("background_audio", OrionAudioService.running)
        .put("capture_active", OrionAudioService.captureActive)
        .put("accessibility", OrionAccessibilityService.isConnected())
        .put("watching", OrionProjectionService.isCapturing)
        .put("sentinel", OrionSentinelService.running)
        .toString()

    @JavascriptInterface
    fun reportModel(model: String) {
        onTrustedPage { onTelemetry(model.take(48), null) }
    }

    @JavascriptInterface
    fun reportMic(active: Boolean) {
        onTrustedPage { onTelemetry("", active) }
    }

    @JavascriptInterface
    fun reportSpeaking(active: Boolean) {
        onTrustedPage { onSpeaking(active) }
    }

    @JavascriptInterface
    fun speak(text: String) {
        onTrustedPage { onSpeak(text) }
    }

    @JavascriptInterface
    fun stopSpeaking() {
        onTrustedPage { onStopSpeaking() }
    }

    @JavascriptInterface
    fun prepareDictation() {
        onTrustedPage { onPrepareDictation() }
    }

    @JavascriptInterface
    fun startDictation() { onTrustedPage { onStartDictation() } }

    @JavascriptInterface
    fun finishDictation() { onTrustedPage { onFinishDictation() } }

    /** Lets the page confirm it is inside the native app and what it can do. */
    @JavascriptInterface
    fun capabilities(): String = if (!trusted()) "" else
        "call,sms,openApp,email,sendEmail,navigate,map,openUrl,share,action,endpointsChanged,deviceStatus,speak,stopSpeaking"
}
