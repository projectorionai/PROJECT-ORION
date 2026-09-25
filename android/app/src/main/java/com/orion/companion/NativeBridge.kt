package com.orion.companion

import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.net.Uri
import android.webkit.JavascriptInterface
import android.widget.Toast

/**
 * Native phone actions exposed to ORION's web UI as `window.OrionNative`.
 *
 * This is what makes the app "more advanced than the HTML": from the same
 * conversation, ORION can place a call, start a text, draft an e-mail, open
 * turn-by-turn navigation, or share — using the phone's own apps, from
 * anywhere he's reachable.
 *
 * SAFETY: every action opens the relevant app *pre-filled* and lets the USER
 * confirm the final tap. We use ACTION_DIAL (never ACTION_CALL) and
 * ACTION_SENDTO (never a silent SMS/e-mail send), so nothing is dialled or
 * sent without a human. No CALL_PHONE / SEND_SMS permission is requested,
 * which also keeps the app off the Play Store's sensitive-permission list.
 *
 * The page feature-detects this object: `if (window.OrionNative) { … }`.
 */
class NativeBridge(
    private val activity: Activity,
    /** Runs on the UI thread after the page reports its endpoints — which it
     *  does straight after pairing and on every token refresh. */
    private val onEndpointsSaved: () -> Unit = {},
) {

    private fun launch(intent: Intent, failMsg: String) {
        activity.runOnUiThread {
            try {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                activity.startActivity(intent)
            } catch (_: ActivityNotFoundException) {
                Toast.makeText(activity, failMsg, Toast.LENGTH_LONG).show()
            } catch (_: Exception) {
                Toast.makeText(activity, failMsg, Toast.LENGTH_LONG).show()
            }
        }
    }

    /** Open the dialer pre-filled with [number]; the user presses call. */
    @JavascriptInterface
    fun call(number: String) {
        launch(
            Intent(Intent.ACTION_DIAL, Uri.parse("tel:${number.trim()}")),
            "No dialer app available.",
        )
    }

    /** Open the SMS composer to [number] with an optional [body]. */
    @JavascriptInterface
    fun sms(number: String, body: String) {
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("smsto:${number.trim()}"))
        if (body.isNotEmpty()) intent.putExtra("sms_body", body)
        launch(intent, "No messaging app available.")
    }

    /** Open the e-mail composer to [to] with an optional [subject]/[body]. */
    @JavascriptInterface
    fun email(to: String, subject: String, body: String) {
        val intent = Intent(Intent.ACTION_SENDTO, Uri.parse("mailto:"))
        if (to.isNotEmpty()) intent.putExtra(Intent.EXTRA_EMAIL, arrayOf(to))
        if (subject.isNotEmpty()) intent.putExtra(Intent.EXTRA_SUBJECT, subject)
        if (body.isNotEmpty()) intent.putExtra(Intent.EXTRA_TEXT, body)
        launch(intent, "No e-mail app available.")
    }

    /** Open turn-by-turn navigation for a free-text [query] (address / place). */
    @JavascriptInterface
    fun navigate(query: String) {
        val q = Uri.encode(query.trim())
        val nav = Intent(Intent.ACTION_VIEW, Uri.parse("google.navigation:q=$q"))
            .setPackage("com.google.android.apps.maps")
        activity.runOnUiThread {
            try {
                nav.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                activity.startActivity(nav)
            } catch (_: Exception) {
                // Fall back to a generic geo query any maps app can handle.
                launch(
                    Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=$q")),
                    "No maps app available.",
                )
            }
        }
    }

    /** Show a place / map for a free-text [query] without starting navigation. */
    @JavascriptInterface
    fun map(query: String) {
        launch(
            Intent(Intent.ACTION_VIEW, Uri.parse("geo:0,0?q=${Uri.encode(query.trim())}")),
            "No maps app available.",
        )
    }

    /** Open an external URL in the phone's browser / relevant app. */
    @JavascriptInterface
    fun openUrl(url: String) {
        launch(Intent(Intent.ACTION_VIEW, Uri.parse(url.trim())), "Can't open that link.")
    }

    /** Offer [text] to the Android share sheet. */
    @JavascriptInterface
    fun share(text: String) {
        val send = Intent(Intent.ACTION_SEND)
            .setType("text/plain")
            .putExtra(Intent.EXTRA_TEXT, text)
        launch(Intent.createChooser(send, "Share via"), "Nothing to share with.")
    }

    /**
     * Store the endpoint list ORION reports at pairing time, so the app can
     * auto-connect (Tailscale name home + away) with no IP ever typed again.
     * Called by the uplink page after a successful pair / token refresh.
     */
    @JavascriptInterface
    fun saveEndpoints(json: String) {
        try {
            Prefs.saveEndpoints(activity, json)
        } catch (_: Exception) {
        }
        // JavaScript-interface calls arrive on a background thread.
        activity.runOnUiThread { onEndpointsSaved() }
    }

    /** Lets the page confirm it is inside the native app and what it can do. */
    @JavascriptInterface
    fun capabilities(): String = "call,sms,email,navigate,map,openUrl,share,saveEndpoints"
}
