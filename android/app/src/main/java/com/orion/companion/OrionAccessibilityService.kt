package com.orion.companion

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.graphics.Rect
import android.os.Handler
import android.os.Bundle
import android.os.Looper
import android.os.SystemClock
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * On-demand access to the active window's semantic UI. This service never
 * records events, streams a screen, or sends data over the network. The native
 * caller is responsible for authorising every request before using this API.
 * Android users must also explicitly enable the service in system settings.
 */
class OrionAccessibilityService : AccessibilityService() {

    private data class SnapshotRef(
        val id: String,
        val revision: Long,
        val windowId: Int,
        val packageName: String,
        val createdAtMs: Long,
        val fingerprints: Map<String, String>,
    )

    private class TraversalState {
        var nodeCount = 0
        var truncated = false
    }

    // All service state and AccessibilityNodeInfo access stays on the main thread.
    private var revision = 0L
    private var lastSnapshot: SnapshotRef? = null
    private data class EmailOperation(
        val request: PhoneActionRules.EmailRequest,
        val canExecute: () -> Boolean,
        val callback: (JSONObject) -> Unit,
        val deadline: Long,
        var clicked: Boolean = false,
    )
    private var emailOperation: EmailOperation? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        activeService = this
        revision++
        lastSnapshot = null
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // Events invalidate snapshots. During one reviewed Gmail operation we
        // also recognise its brief sent/queued receipt; event content is not retained.
        if (event != null) {
            val pending = emailOperation
            if (pending?.clicked == true &&
                event.packageName?.toString() == PhoneActionRules.GMAIL_PACKAGE &&
                event.eventType in setOf(AccessibilityEvent.TYPE_NOTIFICATION_STATE_CHANGED,
                    AccessibilityEvent.TYPE_ANNOUNCEMENT)) {
                val notification = event.text.joinToString(" ").trim()
                if (notification.matches(Regex("(?:Message )?[Ss]ent[.!]?"))) {
                    finishEmail(JSONObject().put("ok", true).put("status", "sent")
                        .put("detail", "Gmail confirmed the reviewed e-mail was sent; recipient delivery is not confirmed"))
                } else if (notification.matches(Regex("(?:Message )?[Qq]ueued[.!]?"))) {
                    finishEmail(JSONObject().put("ok", true).put("status", "queued")
                        .put("detail", "Gmail queued the reviewed e-mail; it has not confirmed sending"))
                }
            }
            revision++
            // A temporary ORION approval overlay can change the event revision
            // without changing the underlying target. act() rechecks the
            // window identity and the selected node fingerprint instead.
        }
    }

    override fun onInterrupt() {
        finishEmail(error("service_interrupted"))
        revision++
        lastSnapshot = null
    }

    override fun onUnbind(intent: Intent?): Boolean {
        finishEmail(error("service_unavailable"))
        if (activeService === this) activeService = null
        lastSnapshot = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        finishEmail(error("service_unavailable"))
        if (activeService === this) activeService = null
        lastSnapshot = null
        super.onDestroy()
    }

    private fun capture(callback: (JSONObject) -> Unit) {
        lastSnapshot = null
        val root = rootInActiveWindow ?: return callback(error("no_active_window"))
        try {
            if (!root.isVisibleToUser) return callback(error("window_not_visible"))
            val packageName = root.packageName?.toString().orEmpty()
            if (packageName.isBlank()) return callback(error("unknown_package"))

            val traversal = TraversalState()
            val fingerprints = HashMap<String, String>()
            val tree = describeNode(root, "0", 0, traversal, fingerprints)
                ?: return callback(error("window_not_visible"))
            val id = UUID.randomUUID().toString()
            lastSnapshot = SnapshotRef(
                id = id,
                revision = revision,
                windowId = root.windowId,
                packageName = packageName,
                createdAtMs = SystemClock.elapsedRealtime(),
                fingerprints = fingerprints,
            )
            callback(JSONObject().apply {
                put("ok", true)
                put("snapshotId", id)
                put("revision", revision)
                put("windowId", root.windowId)
                put("package", packageName)
                put("nodeCount", traversal.nodeCount)
                put("truncated", traversal.truncated)
                put("root", tree)
            })
        } catch (_: Exception) {
            lastSnapshot = null
            callback(error("snapshot_failed"))
        } finally {
            root.recycle()
        }
    }

    private fun describeNode(
        node: AccessibilityNodeInfo,
        id: String,
        depth: Int,
        state: TraversalState,
        fingerprints: MutableMap<String, String>,
    ): JSONObject? {
        if (!node.isVisibleToUser) return null
        if (depth > MAX_DEPTH || state.nodeCount >= MAX_NODES) {
            state.truncated = true
            return null
        }
        state.nodeCount++
        fingerprints[id] = fingerprint(node)

        val bounds = Rect().also(node::getBoundsInScreen)
        val editable = node.isEditable ||
            node.className?.toString()?.endsWith("EditText") == true
        val sensitive = editable || node.isPassword
        val result = JSONObject().apply {
            put("id", id)
            put("class", node.className?.toString()?.take(MAX_CLASS_LENGTH).orEmpty())
            put("viewId", node.viewIdResourceName?.take(160).orEmpty())
            put("bounds", rectJson(bounds))
            put("clickable", node.isClickable)
            put("scrollable", node.isScrollable)
            put("enabled", node.isEnabled)
            put("editable", editable)
            put("password", node.isPassword)
            if (!node.isPassword) safeText(node.hintText).takeIf(String::isNotEmpty)
                ?.let { put("hint", it) }
            if (node.isCheckable) put("checked", node.isChecked)
            if (sensitive) {
                put("valueRedacted", true)
                put("hasValue", !node.text.isNullOrEmpty())
            } else {
                safeText(node.text).takeIf(String::isNotEmpty)?.let { put("text", it) }
                safeText(node.contentDescription).takeIf(String::isNotEmpty)
                    ?.let { put("contentDescription", it) }
                safeText(node.hintText).takeIf(String::isNotEmpty)
                    ?.let { put("hint", it) }
            }
        }

        val children = JSONArray()
        val count = node.childCount
        if (depth == MAX_DEPTH && count > 0) {
            state.truncated = true
        } else {
            val limit = minOf(count, MAX_CHILDREN_PER_NODE)
            if (limit < count) state.truncated = true
            for (index in 0 until limit) {
                if (state.nodeCount >= MAX_NODES) {
                    state.truncated = true
                    break
                }
                val child = try { node.getChild(index) } catch (_: Exception) { null }
                if (child != null) {
                    try {
                        describeNode(child, "$id/$index", depth + 1, state, fingerprints)
                            ?.let { children.put(it) }
                    } finally {
                        child.recycle()
                    }
                }
            }
        }
        if (children.length() > 0) result.put("children", children)
        return result
    }

    /** Click a visible, enabled, clickable node from the latest fresh snapshot. */
    private fun act(
        snapshotId: String,
        nodeId: String,
        direction: String?,
        text: String?,
        callback: (JSONObject) -> Unit,
    ) {
        val snapshot = lastSnapshot
            ?: return callback(error("snapshot_required"))
        if (snapshot.id != snapshotId ||
            SystemClock.elapsedRealtime() - snapshot.createdAtMs !in 0L..SNAPSHOT_TTL_MS
        ) {
            lastSnapshot = null
            return callback(error("stale_snapshot"))
        }
        if (!NODE_ID.matches(nodeId)) return callback(error("invalid_node_id"))
        if (direction != null && direction !in DIRECTIONS) {
            return callback(error("invalid_direction"))
        }
        if (text != null && text.length > 4000) return callback(error("text_too_long"))
        if (snapshot.packageName in BLOCKED_ACTION_PACKAGES ||
            snapshot.packageName == packageName
        ) {
            return callback(error("protected_window"))
        }

        val root = rootInActiveWindow ?: return callback(error("no_active_window"))
        try {
            if (root.windowId != snapshot.windowId ||
                root.packageName?.toString() != snapshot.packageName
            ) {
                lastSnapshot = null
                return callback(error("window_changed"))
            }
            val target = resolveNode(root, nodeId)
                ?: return callback(error("node_not_found"))
            try {
                if (!target.isVisibleToUser || !target.isEnabled ||
                    (direction == null && text == null && !target.isClickable) ||
                    (direction != null && !target.isScrollable)
                ) return callback(error("node_not_actionable"))
                if (snapshot.fingerprints[nodeId] != fingerprint(target)) {
                    lastSnapshot = null
                    return callback(error("node_changed"))
                }
                if (text != null) {
                    val fieldLabel = listOf(target.hintText, target.contentDescription,
                        target.viewIdResourceName).joinToString(" ")
                    if (!target.isEditable || target.isPassword || SENSITIVE_FIELD.containsMatchIn(fieldLabel))
                        return callback(error("protected_text_field"))
                    lastSnapshot = null
                    revision++
                    val accepted = target.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT,
                        Bundle().apply { putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text) })
                    if (!accepted) return callback(error("set_text_rejected"))
                    val updated = target.refresh() && target.text?.toString().orEmpty() == text
                    return callback(JSONObject().put("ok", true).put("action", "set_text")
                        .put("status", if (updated) "updated" else "input_requested")
                        .put("detail", if (updated) "Reviewed text entered in the phone field"
                            else "Android accepted the text input request; check the field"))
                }

                val windowBounds = Rect().also(root::getBoundsInScreen)
                val targetBounds = Rect().also(target::getBoundsInScreen)
                if (!targetBounds.intersect(windowBounds) ||
                    targetBounds.width() < MIN_TARGET_PX ||
                    targetBounds.height() < MIN_TARGET_PX
                ) return callback(error("invalid_bounds"))
                if (((direction == "up" || direction == "down") &&
                        targetBounds.height() < MIN_SCROLL_PX) ||
                    ((direction == "left" || direction == "right") &&
                        targetBounds.width() < MIN_SCROLL_PX)
                ) return callback(error("scroll_target_too_small"))

                val path = gesturePath(targetBounds, direction)
                val duration = if (direction == null) TAP_DURATION_MS else SWIPE_DURATION_MS
                val gesture = GestureDescription.Builder()
                    .addStroke(GestureDescription.StrokeDescription(path, 0, duration))
                    .build()
                lastSnapshot = null // A snapshot authorises at most one gesture.
                revision++
                var replied = false
                fun reply(result: JSONObject) {
                    if (!replied) {
                        replied = true
                        callback(result)
                    }
                }
                val action = if (direction == null) "click" else "scroll"
                val accepted = dispatchGesture(
                    gesture,
                    object : GestureResultCallback() {
                        override fun onCompleted(gestureDescription: GestureDescription) {
                            reply(JSONObject().put("ok", true).put("action", action)
                                .put("status", "gesture_completed")
                                .put("detail", "Android completed the reviewed $action gesture; check the target app for its result"))
                        }

                        override fun onCancelled(gestureDescription: GestureDescription) {
                            reply(error("gesture_cancelled"))
                        }
                    },
                    MAIN,
                )
                if (!accepted) reply(error("gesture_rejected"))
                else MAIN.postDelayed({ reply(error("gesture_result_unavailable")) }, 5_000)
            } finally {
                if (target !== root) target.recycle()
            }
        } catch (_: Exception) {
            callback(error("gesture_failed"))
        } finally {
            root.recycle()
        }
    }

    /** Verifies Gmail's visible recipients and exact content before one send click. */
    private fun sendEmail(request: PhoneActionRules.EmailRequest, canExecute: () -> Boolean,
        callback: (JSONObject) -> Unit) {
        if (emailOperation != null) return callback(error("email_action_busy"))
        if (!PhoneActionRules.validEmailRequest(request)) return callback(error("invalid_email_request"))
        emailOperation = EmailOperation(request, canExecute, callback,
            SystemClock.elapsedRealtime() + 18_000)
        MAIN.postDelayed(::pollEmail, 400)
    }

    private fun finishEmail(result: JSONObject) {
        val pending = emailOperation ?: return
        emailOperation = null
        pending.callback(result)
    }

    private fun pollEmail() {
        val pending = emailOperation ?: return
        if (!pending.canExecute()) { finishEmail(error("command_expired_or_disconnected")); return }
        if (SystemClock.elapsedRealtime() > pending.deadline) {
            finishEmail(if (pending.clicked) JSONObject().put("ok", false).put("status", "send_unverified")
                .put("detail", "Gmail's send control was pressed once, but no sent confirmation appeared. Check Gmail's outbox; ORION will not retry the e-mail.")
            else error("Gmail composer could not verify the exact recipient, subject and body; no e-mail was sent"))
            return
        }
        val root = rootInActiveWindow
        if (root != null) try {
            if (root.packageName?.toString() == PhoneActionRules.GMAIL_PACKAGE) {
                val nodes = ArrayList<AccessibilityNodeInfo>()
                fun collect(node: AccessibilityNodeInfo, depth: Int) {
                    if (nodes.size >= 180 || depth > 18) { node.recycle(); return }
                    nodes.add(node)
                    for (index in 0 until minOf(node.childCount, 50)) {
                        node.getChild(index)?.let { collect(it, depth + 1) }
                    }
                }
                collect(AccessibilityNodeInfo.obtain(root), 0)
                try {
                    val visible = nodes.filter { it.isVisibleToUser }
                    if (pending.clicked) {
                        val receipt = visible.firstOrNull {
                            it.viewIdResourceName?.substringAfterLast('/') == "snackbar_text" &&
                                it.text?.toString()?.trim() in setOf("Sent", "Message sent", "Queued")
                        }?.text?.toString()?.trim()
                        if (receipt != null) finishEmail(JSONObject().put("ok", true)
                            .put("status", if (receipt == "Queued") "queued" else "sent")
                            .put("detail", if (receipt == "Queued") "Gmail queued the reviewed e-mail; sending is not confirmed"
                                else "Gmail confirmed the reviewed e-mail was sent; recipient delivery is not confirmed"))
                    } else {
                        fun field(name: String): List<AccessibilityNodeInfo> = visible.filter {
                            val id = it.viewIdResourceName?.substringAfterLast('/').orEmpty()
                            id == name || (it.isEditable && it.hintText?.toString()?.equals(name, true) == true)
                        }
                        val subject = field("subject").singleOrNull()?.text?.toString()
                        val body = field("body").singleOrNull()?.text?.toString()
                        val recipients = linkedSetOf<String>()
                        val cc = linkedSetOf<String>()
                        val bcc = linkedSetOf<String>()
                        for (node in visible) {
                            val id = node.viewIdResourceName?.substringAfterLast('/').orEmpty()
                            val values = EMAIL.findAll(node.text?.toString().orEmpty() + " " + node.contentDescription?.toString().orEmpty())
                                .map { it.value }.toList()
                            var inheritedRole = ""
                            var ancestor = node.parent
                            var ancestry = 0
                            while (ancestor != null && ancestry++ < 8) {
                                val parentId = ancestor.viewIdResourceName?.substringAfterLast('/').orEmpty()
                                if (parentId == "bcc" || parentId.startsWith("bcc_")) inheritedRole = "bcc"
                                else if (parentId == "cc" || parentId.startsWith("cc_")) inheritedRole = "cc"
                                val next = ancestor.parent
                                ancestor.recycle()
                                ancestor = next
                            }
                            ancestor?.recycle()
                            when {
                                id == "bcc" || id.startsWith("bcc_") || inheritedRole == "bcc" -> bcc.addAll(values)
                                id == "cc" || id.startsWith("cc_") || inheritedRole == "cc" -> cc.addAll(values)
                                id == "to" || id.startsWith("to_") || id.startsWith("peoplekit_") || id == "email_address" -> recipients.addAll(values)
                            }
                        }
                        val sendControls = visible.filter {
                            it.isEnabled && it.isClickable &&
                                (it.viewIdResourceName?.substringAfterLast('/') == "send" ||
                                    it.contentDescription?.toString() in setOf("Send", "Send email"))
                        }
                        val composer = PhoneActionRules.EmailComposer(PhoneActionRules.GMAIL_PACKAGE,
                            recipients, cc, bcc, subject, body, sendControls.size)
                        if (PhoneActionRules.matchesReviewedEmail(pending.request, composer)) {
                            if (!pending.canExecute()) { finishEmail(error("command_expired_or_disconnected")); return }
                            pending.clicked = true // Never click twice, even when the network stalls.
                            if (!sendControls.single().performAction(AccessibilityNodeInfo.ACTION_CLICK))
                                finishEmail(error("Gmail rejected the send action; no retry was attempted"))
                        }
                    }
                } finally { nodes.forEach { it.recycle() } }
            }
        } catch (_: Exception) { finishEmail(error("Gmail verification unavailable; inspect the draft or outbox before retrying")) }
        finally { root?.recycle() }
        if (emailOperation != null) MAIN.postDelayed(::pollEmail, 350)
    }

    private fun resolveNode(root: AccessibilityNodeInfo, nodeId: String): AccessibilityNodeInfo? {
        var current = root
        for (part in nodeId.split('/').drop(1)) {
            val index = part.toIntOrNull() ?: return null
            if (index !in 0 until minOf(current.childCount, MAX_CHILDREN_PER_NODE)) {
                if (current !== root) current.recycle()
                return null
            }
            val child = try { current.getChild(index) } catch (_: Exception) { null }
            if (current !== root) current.recycle()
            if (child == null) return null
            current = child
        }
        return current
    }

    private fun gesturePath(bounds: Rect, direction: String?): Path {
        val path = Path()
        val centerX = bounds.centerX().toFloat()
        val centerY = bounds.centerY().toFloat()
        if (direction == null) {
            path.moveTo(centerX, centerY)
            return path
        }
        val left = bounds.left + bounds.width() * 0.2f
        val right = bounds.right - bounds.width() * 0.2f
        val top = bounds.top + bounds.height() * 0.2f
        val bottom = bounds.bottom - bounds.height() * 0.2f
        when (direction) {
            "up" -> { path.moveTo(centerX, bottom); path.lineTo(centerX, top) }
            "down" -> { path.moveTo(centerX, top); path.lineTo(centerX, bottom) }
            "left" -> { path.moveTo(right, centerY); path.lineTo(left, centerY) }
            "right" -> { path.moveTo(left, centerY); path.lineTo(right, centerY) }
        }
        return path
    }

    private fun fingerprint(node: AccessibilityNodeInfo): String {
        val rect = Rect().also(node::getBoundsInScreen)
        val sensitive = node.isPassword || node.isEditable
        return listOf(
            node.className?.toString().orEmpty(),
            node.viewIdResourceName.orEmpty(),
            rect.left, rect.top, rect.right, rect.bottom,
            node.isClickable, node.isScrollable, node.isEnabled,
            node.isEditable, node.isPassword,
            if (node.isCheckable) node.isChecked else null,
            if (sensitive) "[redacted]" else safeText(node.text),
            if (sensitive) "[redacted]" else safeText(node.contentDescription),
        ).joinToString("|")
    }

    companion object {
        private const val MAX_NODES = 120
        private const val MAX_DEPTH = 12
        private const val MAX_CHILDREN_PER_NODE = 50
        private const val MAX_CLASS_LENGTH = 100
        private const val MAX_TEXT_LENGTH = 160
        private const val SNAPSHOT_TTL_MS = 10_000L
        private const val MIN_TARGET_PX = 12
        private const val MIN_SCROLL_PX = 80
        private const val TAP_DURATION_MS = 75L
        private const val SWIPE_DURATION_MS = 350L
        private val MAIN = Handler(Looper.getMainLooper())
        private val NODE_ID = Regex("0(?:/[0-9]{1,2}){0,$MAX_DEPTH}")
        private val DIRECTIONS = setOf("up", "down", "left", "right")
        private val BLOCKED_ACTION_PACKAGES = setOf(
            "com.android.systemui",
            "com.android.settings",
            "com.android.permissioncontroller",
            "com.google.android.permissioncontroller",
            "com.android.packageinstaller",
            "com.google.android.packageinstaller",
        )
        private val SPACES = Regex("\\s+")
        private val URL = Regex("https?://\\S+", RegexOption.IGNORE_CASE)
        private val EMAIL = Regex("[A-Z0-9._%+-]+@[A-Z0-9.-]+\\.[A-Z]{2,}", RegexOption.IGNORE_CASE)
        private val CARD = Regex("(?<!\\d)(?:\\d[ -]?){13,19}(?!\\d)")
        private val PHONE = Regex("(?<!\\w)\\+?\\d[\\d(). -]{7,}\\d(?!\\w)")
        private val CODE = Regex(
            "\\b(?:otp|one.time|verification|security code|passcode|pin)\\b.{0,32}\\b\\d{4,8}\\b",
            RegexOption.IGNORE_CASE,
        )
        private val TOKEN = Regex("\\b(?:Bearer\\s+[A-Za-z0-9._-]{12,}|sk-[A-Za-z0-9_-]{12,})\\b")
        private val STANDALONE_CODE = Regex("(?:\\d[ -]?){4,8}")
        private val SENSITIVE_FIELD = Regex("password|passcode|one.?time|verification|security.?code|otp|\\bpin\\b|card.?number|cvv|cvc", RegexOption.IGNORE_CASE)

        @Volatile
        private var activeService: OrionAccessibilityService? = null

        fun isConnected(): Boolean = activeService != null

        /** Callback always runs on the Android main thread. */
        fun requestSnapshot(callback: (JSONObject) -> Unit) {
            MAIN.post {
                val service = activeService
                if (service == null) callback(error("service_unavailable"))
                else service.capture(callback)
            }
        }

        /** Requires a node ID and snapshot ID returned by requestSnapshot. */
        fun clickNode(snapshotId: String, nodeId: String, callback: (JSONObject) -> Unit) {
            MAIN.post {
                val service = activeService
                if (service == null) callback(error("service_unavailable"))
                else service.act(snapshotId, nodeId, null, null, callback)
            }
        }

        /** Scroll a visible scrollable node: up, down, left, or right. */
        fun scrollNode(
            snapshotId: String,
            nodeId: String,
            direction: String,
            callback: (JSONObject) -> Unit,
        ) {
            MAIN.post {
                val service = activeService
                if (service == null) callback(error("service_unavailable"))
                else service.act(snapshotId, nodeId, direction, null, callback)
            }
        }

        fun setNodeText(snapshotId: String, nodeId: String, text: String, callback: (JSONObject) -> Unit) {
            MAIN.post {
                val service = activeService
                if (service == null) callback(error("service_unavailable"))
                else service.act(snapshotId, nodeId, null, text, callback)
            }
        }

        /** For a deliberate local user action; Android acceptance is not app success. */
        fun navigate(action: String, callback: (JSONObject) -> Unit) {
            MAIN.post {
                val service = activeService
                val code = when (action) { "back" -> GLOBAL_ACTION_BACK; "home" -> GLOBAL_ACTION_HOME; else -> null }
                if (service == null) callback(error("service_unavailable"))
                else if (code == null) callback(error("unsupported_navigation"))
                else if (service.performGlobalAction(code)) callback(JSONObject().put("ok", true)
                    .put("action", action).put("status", "navigation_requested")
                    .put("detail", "Android accepted the phone $action action"))
                else callback(error("navigation_rejected"))
            }
        }

        internal fun sendReviewedEmail(request: PhoneActionRules.EmailRequest,
            canExecute: () -> Boolean, callback: (JSONObject) -> Unit) {
            MAIN.post {
                val service = activeService
                if (service == null) callback(error("service_unavailable"))
                else service.sendEmail(request, canExecute, callback)
            }
        }

        private fun error(reason: String) = JSONObject().put("ok", false).put("error", reason)

        private fun rectJson(bounds: Rect) = JSONObject()
            .put("left", bounds.left)
            .put("top", bounds.top)
            .put("right", bounds.right)
            .put("bottom", bounds.bottom)

        private fun safeText(value: CharSequence?): String {
            if (value == null || value.isEmpty()) return ""
            val bounded = value.subSequence(0, minOf(value.length, MAX_TEXT_LENGTH * 4)).toString()
            var safe = SPACES.replace(bounded, " ")
            safe = URL.replace(safe, "[url]")
            safe = EMAIL.replace(safe, "[email]")
            safe = CARD.replace(safe, "[number]")
            safe = PHONE.replace(safe, "[phone]")
            safe = CODE.replace(safe, "[code]")
            safe = TOKEN.replace(safe, "[token]")
            if (STANDALONE_CODE.matches(safe.trim())) return "[code]"
            return safe.trim().take(MAX_TEXT_LENGTH)
        }
    }
}
