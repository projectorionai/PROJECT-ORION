package com.orion.companion

import android.Manifest
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.projection.MediaProjectionManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.PixelCopy
import android.widget.*
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.bottomsheet.BottomSheetDialog
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicReference

/** Native phone terminal, microphone and device tools; no PC needed after provisioning. */
class PhoneConsoleActivity : AppCompatActivity() {
    private val main = Handler(Looper.getMainLooper())
    private var pcControls: RemotePcControls? = null
    private val worker = Executors.newSingleThreadExecutor { job -> Thread(job, "orion-phone-turn").apply { isDaemon = true } }
    private val captureWorker = Executors.newSingleThreadExecutor { job ->
        Thread(job, "orion-phone-screenshot").apply { isDaemon = true }
    }
    private val generation = AtomicLong()
    private val historyGeneration = AtomicLong()
    private lateinit var brain: PhoneBrain
    private lateinit var voice: PhoneVoice
    private lateinit var dictation: NativeDictation
    private lateinit var title: TextView
    private lateinit var status: TextView
    private lateinit var heard: TextView
    private lateinit var waveform: TactileWaveformView
    private lateinit var avatar: PhoneAvatarView
    private lateinit var messages: LinearLayout
    private lateinit var emptyConversation: TextView
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var send: Button
    private lateinit var microphone: Button
    private lateinit var stop: Button
    private var busy = false
    private var errorLatched = false
    private var voiceStatusLatched = false
    private var resumed = false
    private var restoreListening = false
    private var permissionCallback: ((Boolean) -> Unit)? = null
    private var review: AlertDialog? = null
    private val activeActions = mutableSetOf<String>()
    private var shownHistoryId = ""
    private var restoringHistory = false
    private var pendingScreenshotPath: String? = null
    private var captureBusy = false
    /** The PC's face and emotions, while it is reachable (LAN first). */
    private var pcLink: PcFaceLink? = null
    /** Requests made with no internet, sent in order when it returns. */
    private val offlineQueue = OfflineQueue()
    private val actionHandler: (PhoneActionCoordinator.Request) -> Unit = { request -> perform(request) }
    private val permissions = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val callback = permissionCallback; permissionCallback = null; callback?.invoke(granted)
    }
    private val screenshotDocument = registerForActivityResult(ActivityResultContracts.CreateDocument("image/png")) { uri ->
        val path = pendingScreenshotPath
        pendingScreenshotPath = null
        if (path == null) { captureBusy = false; return@registerForActivityResult }
        if (uri == null) { File(path).delete(); captureBusy = false; return@registerForActivityResult }
        captureWorker.execute {
            val saved = runCatching {
                File(path).inputStream().use { source ->
                    val output = contentResolver.openOutputStream(uri, "w")
                        ?: throw IOException("The chosen screenshot file could not be opened.")
                    output.use(source::copyTo)
                }
            }.isSuccess
            File(path).delete()
            main.post { if (!isDestroyed) {
                captureBusy = false
                state(if (saved) "Screenshot saved" else "Screenshot failed",
                    if (saved) "" else "Choose another location and try again.")
            } }
        }
    }
    private val projection = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK && result.data != null) {
            runCatching { OrionProjectionService.start(this, result.resultCode, result.data!!) }
                .onSuccess { awaitScreenCapture(0) }
                .onFailure { state("Screen access failed", "Android could not start screen sharing. Please try again.") }
        } else state("Screen access off", "Screen sharing was not approved.")
    }
    private val meter = object : Runnable {
        override fun run() {
            if (isDestroyed || !resumed) return
            for (event in PhoneListeningService.drainEvents()) {
                val item = runCatching { JSONObject(event) }.getOrNull() ?: continue
                when (item.optString("state")) {
                    "transcript" -> showHeard(item.optString("text"))
                    "reply" -> { loadHistory(); state("Speaking", "") }
                    "level" -> waveform.amplitude = item.optDouble("level", 0.0).toFloat()
                    "error" -> showError(item.optString("text"))
                    "voice_error" -> showVoiceProblem(item.optString("text"))
                    "listening" -> state("Listening", item.optString("text"))
                    "wake_required" -> state("Say ‘ORION’", item.optString("text"))
                    "wake_ready" -> { clearError(); state("Listening", item.optString("text")) }
                    "recovered" -> { clearError(); state("Listening", item.optString("text")) }
                    "speaking" -> state("Speaking", item.optString("text"))
                    "connecting", "transcribing", "thinking" -> state("Working", item.optString("text"))
                }
            }
            waveform.watching = PhoneListeningService.running || dictation.busy
            waveform.speaking = PhoneVoice.anySpeaking() || RemoteSpeech.anySpeaking()
            if (waveform.speaking) waveform.amplitude = PhoneVoice.outputLevel
            else if (!waveform.watching) waveform.amplitude = 0f
            val listening = waveform.watching
            val speaking = waveform.speaking
            val thinking = busy && !speaking
            val link = pcLink
            val linked = link?.connected == true
            val online = PhoneNetwork.online(this@PhoneConsoleActivity)
            val phoneState = PhoneAvatarPolicy.state(listening, thinking, speaking, offline = !online && !linked)
            avatar.link(linked)
            if (linked && link != null) {
                avatar.show(PhoneAvatarPolicy.linked(link.state, phoneState),
                    speaking || (phoneState == "STANDBY" && link.speaking),
                    if (speaking || listening) waveform.amplitude else link.amplitude)
            } else avatar.show(phoneState, speaking, waveform.amplitude)
            if (online && !busy && offlineQueue.isNotEmpty() && !errorLatched) sendNextQueued()
            stop.visibility = if (busy || PhoneVoice.anySpeaking() || RemoteSpeech.anySpeaking()) View.VISIBLE else View.GONE
            main.postDelayed(this, 300)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        PhoneAiProfile.setPhoneMode(this, true)
        window.statusBarColor = Color.rgb(8, 8, 10)
        window.navigationBarColor = Color.rgb(8, 8, 10)
        buildScreen()
        pendingScreenshotPath = savedInstanceState?.getString("pending_screenshot")
        captureBusy = pendingScreenshotPath != null
        val activeScreenshot = pendingScreenshotPath
        captureWorker.execute { cacheDir.listFiles()?.filter {
            it.name.startsWith("orion-screenshot-") &&
                it.absolutePath != activeScreenshot &&
                System.currentTimeMillis() - it.lastModified() > 24 * 60 * 60_000L
        }?.forEach(File::delete) }
        brain = PhoneBrain(this)
        voice = PhoneVoice(this, onSpeaking = { speaking -> main.post {
            if (speaking) { voiceStatusLatched = false; state("Speaking", "") }
            else if (!voiceStatusLatched)
                state(if (PhoneListeningService.running) "Listening" else "Voice", "")
        } }, onStatus = { detail -> main.post {
            if (!errorLatched) {
                val natural = if (detail == "Charon") "" else detail
                    .replace("Charon", "The usual voice", ignoreCase = true)
                    .replace("Gemini", "The assistant", ignoreCase = true)
                voiceStatusLatched = natural.startsWith("Voice unavailable") ||
                    natural.startsWith("Voice connection lost")
                if (voiceStatusLatched) title.text = "Voice"
                status.text = natural
                status.visibility = if (natural.isBlank()) View.GONE else View.VISIBLE
            }
        } })
        dictation = NativeDictation(this, onEvent = { event ->
            when (event.optString("state")) {
                "listening" -> { state("Listening", "Speak naturally. Tap the microphone to finish."); microphone.text = "Finish" }
                "level" -> waveform.amplitude = event.optDouble("level", 0.0).toFloat()
                "transcribing" -> state("Understanding", "")
                "transcript" -> { val words = event.optString("text"); showHeard(words); submit(words, fromMicrophone = true) }
                "error" -> showError(event.optString("text"))
                "finished" -> { microphone.text = "Mic"; waveform.amplitude = 0f
                    if (!busy && !errorLatched) state(if (PhoneListeningService.running) "Listening" else "Voice", "")
                    if (restoreListening && resumed) { restoreListening = false; startListening() } }
            }
        }, transcriber = { wav -> brain.transcribe(wav) })
        if (!PhoneAiProfile.hasKey(this) && !PhoneHostedProfile.ready(this)) showFirstRunSetup()
        else if (PhoneListeningService.requested(this) && !PhoneListeningService.running)
            main.post { if (!isDestroyed) startListening() }
        handleSentinel(intent)
    }

    private fun awaitScreenCapture(attempt: Int) {
        if (isDestroyed) return
        if (OrionProjectionService.isCapturing)
            state("Screen access on", "Ask ORION about the shared screen. Stop sharing in Options when finished.")
        else if (attempt < 15) main.postDelayed({ awaitScreenCapture(attempt + 1) }, 200)
        else state("Screen access failed", "Android did not start screen sharing. Please try again.")
    }

    private fun buildScreen() {
        val page = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), dp(10))
            setBackgroundColor(Color.rgb(8, 8, 10))
        }
        val header = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(View(this).apply { background = rounded(Color.rgb(211, 47, 47), dp(2)) },
            LinearLayout.LayoutParams(dp(3), dp(25)).apply { rightMargin = dp(11) })
        header.addView(text("ORION", 23f).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            letterSpacing = 0.12f
            setTextColor(Color.WHITE)
        }, LinearLayout.LayoutParams(0, dp(56), 1f))
        header.addView(button("Options") { options() }, LinearLayout.LayoutParams(dp(100), dp(46)))
        page.addView(header, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(6) })
        // His day, one tap away and working with the PC off: Calendar, Focus, Fitness.
        val day = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        listOf("Calendar" to DayActivity.CALENDAR, "Focus" to DayActivity.FOCUS,
            "Fitness" to DayActivity.FITNESS).forEachIndexed { index, (name, tab) ->
            day.addView(text(name, 13f).apply {
                gravity = Gravity.CENTER
                setTextColor(Color.rgb(223, 226, 235))
                background = rounded(Color.rgb(22, 22, 26), dp(17), Color.rgb(58, 42, 47))
                isClickable = true
                isFocusable = true
                setOnClickListener { DayActivity.open(this@PhoneConsoleActivity, tab) }
            }, LinearLayout.LayoutParams(0, dp(36), 1f).apply { if (index > 0) leftMargin = dp(8) })
        }
        page.addView(day, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
        val listening = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(8), dp(13), dp(8))
            background = GradientDrawable(GradientDrawable.Orientation.TL_BR,
                intArrayOf(Color.rgb(27, 21, 24), Color.rgb(17, 17, 20))).apply {
                cornerRadius = dp(16).toFloat()
                setStroke(dp(1), Color.rgb(65, 43, 47))
            }
            elevation = dp(2).toFloat()
        }
        avatar = PhoneAvatarView(this).apply {
            contentDescription = "ORION's face"
            amplitudeSource = {
                val link = pcLink
                when {
                    PhoneVoice.anySpeaking() -> PhoneVoice.outputLevel
                    waveform.watching -> waveform.amplitude
                    link?.connected == true -> link.amplitude
                    else -> waveform.amplitude
                }
            }
        }
        listening.addView(avatar, LinearLayout.LayoutParams(dp(118), dp(126)).apply { rightMargin = dp(11) })
        val listeningText = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        title = text("Voice", 20f).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(Color.WHITE)
        }
        status = text("", 12f).apply {
            setTextColor(Color.rgb(174, 174, 183))
            setLineSpacing(dp(2).toFloat(), 1f)
            visibility = View.GONE
        }
        heard = text("", 12f).apply {
            setTextColor(Color.rgb(242, 230, 232))
            visibility = View.GONE
            maxLines = 3
            ellipsize = android.text.TextUtils.TruncateAt.END
        }
        waveform = TactileWaveformView(this)
        listeningText.addView(title, LinearLayout.LayoutParams(-1, -2))
        listeningText.addView(waveform, LinearLayout.LayoutParams(-1, dp(48)).apply { topMargin = dp(2) })
        listeningText.addView(status)
        listeningText.addView(heard, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(4) })
        listening.addView(listeningText, LinearLayout.LayoutParams(0, -2, 1f))
        page.addView(listening, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
        messages = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        emptyConversation = text("Ask ORION anything.", 13f).apply {
            setTextColor(Color.rgb(145, 145, 152))
            setPadding(dp(4), dp(10), dp(4), dp(10))
        }
        messages.addView(emptyConversation)
        scroll = ScrollView(this).apply { addView(messages); isFillViewport = true }
        page.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        stop = button("Stop") { cancelTurn() }.apply { visibility = View.GONE }
        page.addView(stop, LinearLayout.LayoutParams(-1, dp(44)))
        input = EditText(this).apply {
            hint = "Ask or give an instruction…"
            setTextColor(Color.WHITE)
            setHintTextColor(Color.rgb(145, 145, 152))
            textSize = 15f
            maxLines = 3
            background = rounded(Color.rgb(26, 26, 30), dp(14), Color.rgb(55, 55, 62))
            setPadding(dp(14), dp(10), dp(14), dp(10))
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        }
        val actions = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        microphone = button("Mic") { dictate() }
        send = button("Send", true) { val value = input.text.toString(); if (value.isNotBlank()) { input.setText(""); submit(value) } }
        actions.addView(microphone, LinearLayout.LayoutParams(dp(72), dp(52)).apply { rightMargin = dp(7) })
        actions.addView(input, LinearLayout.LayoutParams(0, -2, 1f).apply { rightMargin = dp(7) })
        actions.addView(send, LinearLayout.LayoutParams(dp(72), dp(52)))
        page.addView(actions, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(8) })
        setContentView(page)
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun rounded(colour: Int, radius: Int = dp(14), border: Int? = null) = GradientDrawable().apply {
        setColor(colour)
        cornerRadius = radius.toFloat()
        if (border != null) setStroke(dp(1), border)
    }
    private fun text(value: String, size: Float) = TextView(this).apply { this.text = value; textSize = size; setTextColor(Color.rgb(223, 226, 235)); setPadding(0, dp(3), 0, dp(3)) }
    private fun button(label: String, accent: Boolean = false, action: () -> Unit) = Button(this).apply {
        text = label
        isAllCaps = false
        minHeight = dp(46)
        setTextColor(Color.WHITE)
        backgroundTintList = android.content.res.ColorStateList.valueOf(
            if (accent) Color.rgb(164, 32, 42) else Color.rgb(42, 42, 48))
        setOnClickListener { action() }
    }
    private fun clearError() { errorLatched = false; voiceStatusLatched = false }
    private fun state(label: String, detail: String) {
        if (errorLatched) return
        title.text = label
        status.text = detail
        status.visibility = if (detail.isBlank()) View.GONE else View.VISIBLE
    }
    private fun showError(detail: String) {
        errorLatched = true
        title.text = "Request failed"
        status.text = detail.ifBlank { "Please try again. If this continues, check the connection in Options." }
        status.visibility = View.VISIBLE
    }
    private fun showVoiceProblem(detail: String) {
        if (errorLatched) return
        voiceStatusLatched = true
        title.text = "Voice"
        status.text = detail
        status.visibility = View.VISIBLE
    }
    private fun showHeard(words: String) {
        heard.text = "Heard: $words"
        heard.visibility = if (words.isBlank()) View.GONE else View.VISIBLE
    }
    private fun add(role: String, value: String): TextView {
        emptyConversation.visibility = View.GONE
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(15), dp(9), dp(15), dp(12))
            background = rounded(if (role == "user") Color.rgb(37, 37, 42) else Color.rgb(20, 20, 23),
                dp(14), if (role == "model") Color.rgb(83, 39, 45) else Color.rgb(48, 48, 54))
            elevation = dp(1).toFloat()
        }
        card.addView(text(if (role == "user") "YOU" else if (role == "meta") "NOTICE" else "ORION", 10f).apply {
            typeface = Typeface.DEFAULT_BOLD
            letterSpacing = 0.12f
            setTextColor(if (role == "user") Color.rgb(172, 172, 180) else Color.rgb(211, 47, 47))
        })
        val node = text(value, 15f).apply {
            setTextColor(Color.rgb(242, 242, 244))
            setTextIsSelectable(true)
            setLineSpacing(dp(3).toFloat(), 1f)
        }
        card.addView(node)
        messages.addView(card, LinearLayout.LayoutParams(-1, -2).apply {
            bottomMargin = dp(10)
            if (role == "user") leftMargin = dp(42) else rightMargin = dp(15)
        })
        while (messages.childCount > 100) messages.removeViewAt(0)
        if (!restoringHistory) scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
        return node
    }
    private fun addReceipt(action: JSONObject, receipt: JSONObject) {
        emptyConversation.visibility = View.GONE
        val success = receipt.optBoolean("ok")
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(15), dp(11), dp(15), dp(11))
            background = rounded(if (success) Color.rgb(28, 28, 32) else Color.rgb(48, 23, 28),
                dp(14), if (success) Color.rgb(57, 57, 63) else Color.rgb(117, 42, 52))
            elevation = dp(2).toFloat()
        }
        val operation = action.optString("kind").replace('_', ' ').uppercase()
        card.addView(text("${if (success) "DONE" else "ACTION FAILED"} · $operation", 11f)
            .apply { typeface = Typeface.DEFAULT_BOLD })
        if (success && action.optString("kind") == "device_status" && receipt.has("connection")) {
            val percentage = receipt.optInt("battery_percent", -1)
            val battery = if (percentage in 0..100) "$percentage%" else "Unknown"
            card.addView(text("Battery  $battery", 21f).apply { setTextColor(Color.WHITE) })
            if (percentage in 0..100) card.addView(ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 100
                progress = percentage
                progressTintList = android.content.res.ColorStateList.valueOf(Color.rgb(211, 47, 65))
                progressBackgroundTintList = android.content.res.ColorStateList.valueOf(Color.rgb(66, 66, 73))
            }, LinearLayout.LayoutParams(-1, dp(6)).apply { topMargin = dp(5); bottomMargin = dp(8) })
            val flags = buildList {
                add(receipt.optString("connection", "Offline"))
                if (receipt.optBoolean("charging")) add("Charging")
                if (receipt.optBoolean("power_saving")) add("Power saving")
            }
            card.addView(text(flags.joinToString("  ·  "), 12f))
        } else card.addView(text(receipt.optString("detail", receipt.optString("status", "No receipt was returned.")), 13f))
        messages.addView(card, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        while (messages.childCount > 100) messages.removeViewAt(0)
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }
    private fun loadHistory() {
        if (busy || isDestroyed) return
        val request = historyGeneration.incrementAndGet()
        runCatching { worker.execute {
            val entries = PhoneConversationStore.entries(this)
            main.post {
                if (isDestroyed || busy || request != historyGeneration.get()) return@post
                if (entries.length() == 0) {
                    if (shownHistoryId.isNotEmpty()) {
                        shownHistoryId = ""
                        messages.removeAllViews()
                        messages.addView(emptyConversation)
                    }
                    emptyConversation.visibility = View.VISIBLE
                    return@post
                }
                val latest = entries.getJSONObject(entries.length() - 1).optString("id")
                if (latest == shownHistoryId) return@post
                messages.removeAllViews()
                messages.addView(emptyConversation)
                restoringHistory = true
                try {
                    for (i in maxOf(0, entries.length() - 100) until entries.length()) {
                        val item = entries.getJSONObject(i)
                        add(item.optString("role"), item.optString("text"))
                    }
                } finally { restoringHistory = false }
                shownHistoryId = latest
                scroll.post { if (!isDestroyed) scroll.fullScroll(View.FOCUS_DOWN) }
            }
        } }
    }
    private fun submit(value: String, fromMicrophone: Boolean = false, queued: Boolean = false) {
        if (busy) { state("A request is already running", "Use Stop before starting another request."); return }
        clearError()
        val message = value.trim().take(16000)
        if (message.isEmpty()) return
        if (!queued) avatar.feel(message, fromUser = true)
        // "Open my calendar", "start a focus block", "log a workout": the page
        // itself, at once and offline, never a model's description of it.
        DayPlanner.pageFor(message)?.let { tab ->
            add("user", message)
            DayActivity.open(this, tab)
            return
        }
        if (!PhoneNetwork.online(this)) {
            // No internet: the hosted AI cannot answer. What the phone knows
            // itself is answered now; anything else waits for the connection.
            val local = OfflineAnswers.answer(this, message)
            if (local != null) {
                add("user", message); add("model", local); avatar.feel(local, fromUser = false)
                state("Offline", "Answered on this phone.")
                if (getPreferences(MODE_PRIVATE).getBoolean("voice", true)) voice.speak(local)
                return
            }
            if (!queued) {
                add("user", message)
                val kept = offlineQueue.add(message)
                add("meta", if (kept) "Offline. This request will be sent when the phone is back online."
                    else "Offline, and ${OfflineQueue.CAPACITY} requests are already waiting. This one was not saved.")
            }
            state("Offline", "${offlineQueue.size} request(s) waiting for a connection.")
            return
        }
        if (!PhoneHostedProfile.ready(this)) {
            showError("Set up the selected text AI provider in Options."); return
        }
        if (!fromMicrophone) PhonePerformance.route(this, "Typed")
        SpeechPlaybackEpoch.begin()
        historyGeneration.incrementAndGet()
        PhoneVoice.stopActiveAll(); RemoteSpeech.stopAll()
        main.post { if (!isDestroyed) voice.prepare() }
        busy = true; send.isEnabled = false
        val id = generation.incrementAndGet()
        if (!queued) add("user", message)
        val replyNode = add("model", "Thinking…")
        state("Thinking", "")
        // The phone engine works without a Gemini key; cloud failure can also
        // fall back to it for customers using another text provider.
        val speakReply = getPreferences(MODE_PRIVATE).getBoolean("voice", true)
        val streamSpeech = speakReply && SpeechText.mayStreamAnswer(message)
        worker.execute {
            var reply: String
            var succeeded = true
            val speechDraft = SpeechText.Stream()
            var spokenTurn: PhoneVoice.IncrementalTurn? = null
            val latestStreamText = AtomicReference("")
            val streamUpdatePending = AtomicBoolean(false)
            val streamFinished = AtomicBoolean(false)
            try {
                PhoneConversationStore.append(this, "user", message)
                val assembled = StringBuilder()
                reply = brain.reply(message, onDelta = { delta ->
                    assembled.append(delta)
                    latestStreamText.set(assembled.toString())
                    if (streamUpdatePending.compareAndSet(false, true)) {
                        main.postDelayed({
                            streamUpdatePending.set(false)
                            if (!streamFinished.get() && id == generation.get() && !isDestroyed) {
                                val follow = scroll.scrollY + scroll.height >= messages.height - dp(64)
                                replyNode.text = latestStreamText.get()
                                if (follow) scroll.post { if (!isDestroyed) scroll.fullScroll(View.FOCUS_DOWN) }
                            }
                        }, 45)
                    }
                    if (streamSpeech && id == generation.get()) {
                        for (phrase in speechDraft.add(delta)) {
                            val turn = spokenTurn ?: voice.beginIncremental().also { spokenTurn = it }
                            turn.enqueue(phrase)
                        }
                    }
                }, execute = { action, report -> PhoneActionCoordinator.request(this, action) { receipt ->
                    main.post { if (id == generation.get() && !isDestroyed) addReceipt(action, receipt) }
                    report(receipt)
                } })
                if (streamSpeech && spokenTurn != null && id == generation.get()) {
                    if (reply.startsWith(assembled.toString()))
                        speechDraft.add(reply.substring(assembled.length)).forEach { spokenTurn?.enqueue(it) }
                    speechDraft.finish().takeIf { it.isNotEmpty() }?.let { spokenTurn?.enqueue(it) }
                    spokenTurn?.finish()
                }
            } catch (failure: Exception) {
                succeeded = false
                spokenTurn?.cancel()
                reply = PhoneBrain.failureText(failure)
            }
            streamFinished.set(true)
            if (id == generation.get()) {
                if (succeeded) runCatching { PhoneConversationStore.append(this, "model", reply) }
                val latestHistoryId = runCatching {
                    val entries = PhoneConversationStore.entries(this)
                    if (entries.length() > 0) entries.getJSONObject(entries.length() - 1).optString("id") else ""
                }.getOrDefault("")
                main.post { if (id == generation.get() && !isDestroyed) {
                    val follow = scroll.scrollY + scroll.height >= messages.height - dp(64)
                    busy = false; send.isEnabled = true; replyNode.text = reply
                    if (follow) scroll.post { if (!isDestroyed) scroll.fullScroll(View.FOCUS_DOWN) }
                    shownHistoryId = latestHistoryId
                    if (succeeded) {
                        avatar.feel(reply, fromUser = false)
                        state(if (PhoneListeningService.running) "Listening" else "Voice", "")
                        if (speakReply && spokenTurn == null)
                            voice.speak(reply)
                    } else showError(reply)
                } }
            }
        }
    }
    /** Back online: send the oldest saved request (one at a time, in order).
     *  Phone actions in it still need their usual native review. */
    private fun sendNextQueued() {
        val next = offlineQueue.poll() ?: return
        add("meta", "Back online. Sending your saved request.")
        submit(next, queued = true)
    }

    private fun cancelTurn() {
        offlineQueue.clear()
        generation.incrementAndGet(); brain.cancel(); dictation.cancel()
        PhoneActionCoordinator.cancelAll(this); review?.dismiss(); review = null
        PhoneVoice.stopAll(); RemoteSpeech.stopAll()
        busy = false; send.isEnabled = true; microphone.text = "Mic"
        clearError()
        state("Stopped", "Ready for your next phone instruction.")
    }
    private fun changeAiConfiguration(change: () -> Unit) {
        cancelTurn()
        PhoneListeningService.updateAiConfiguration(change)
    }
    private fun permission(name: String, callback: (Boolean) -> Unit) {
        if (ContextCompat.checkSelfPermission(this, name) == PackageManager.PERMISSION_GRANTED) callback(true)
        else if (permissionCallback != null) callback(false)
        else { permissionCallback = callback; permissions.launch(name) }
    }
    /** No internet, so the hosted transcription cannot run: hand the
     *  microphone to Android's own recogniser, asking for its on-device
     *  model. Whether that works offline depends on the phone's installed
     *  offline speech languages; if it cannot, say so plainly. */
    private val offlineSpeech = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val words = result.data?.getStringArrayListExtra(android.speech.RecognizerIntent.EXTRA_RESULTS)
            ?.firstOrNull()?.trim().orEmpty()
        if (result.resultCode == Activity.RESULT_OK && words.isNotEmpty()) { showHeard(words); submit(words, fromMicrophone = true) }
        else state("Offline", "Nothing was recognised. Type the request, or install an offline speech language in Android settings.")
    }
    private fun offlineDictation() {
        val intent = Intent(android.speech.RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(android.speech.RecognizerIntent.EXTRA_LANGUAGE_MODEL, android.speech.RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(android.speech.RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            .putExtra(android.speech.RecognizerIntent.EXTRA_PROMPT, "ORION is offline - speak your request")
        try {
            PhoneVoice.stopActiveAll(); RemoteSpeech.stopAll()
            offlineSpeech.launch(intent)
        } catch (_: android.content.ActivityNotFoundException) {
            state("Offline", "This phone has no offline speech recogniser. Type the request instead.")
        }
    }
    private fun dictate() {
        if (dictation.busy) { dictation.finish(); return }
        if (busy) { state("A request is running", "Use Stop before dictating a new instruction."); return }
        clearError()
        if (!PhoneNetwork.online(this)) { offlineDictation(); return }
        if (!PhoneAiProfile.hasKey(this)) { provision(); return }
        PhoneVoice.stopActiveAll(); RemoteSpeech.stopAll()
        OfflineNavigationService.interruptSpeech()
        permission(Manifest.permission.RECORD_AUDIO) { granted ->
            if (!granted) { state("Microphone permission required", "Allow the microphone in Android settings, then retry."); return@permission }
            restoreListening = PhoneListeningService.running
            PhoneListeningService.stop(this); OrionAudioService.stop(this)
            fun wait(attempt: Int) {
                if (!resumed || isDestroyed) return
                if (!PhoneListeningService.captureActive && !OrionAudioService.captureActive) dictation.start()
                else if (attempt < 20) main.postDelayed({ wait(attempt + 1) }, 75)
                else state("Microphone is busy", "Stop other recording apps and try again.")
            }
            wait(0)
        }
    }
    private fun startListening() {
        clearError()
        if (!PhoneAiProfile.hasKey(this)) { provision(); return }
        permission(Manifest.permission.RECORD_AUDIO) { granted ->
            if (granted) {
                val start = {
                    runCatching { PhoneListeningService.start(this) }
                        .onFailure { state("Listening could not start", "Reopen ORION and check microphone permission.") }
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU)
                    permission(Manifest.permission.POST_NOTIFICATIONS) { _ -> start() }
                else start()
            } else state("Microphone permission required", "Allow microphone access to start background listening.")
        }
    }
    private fun perform(request: PhoneActionCoordinator.Request) {
        if (!activeActions.add(request.id)) return
        val action = JSONObject(request.action.toString())
        val brainRequest = action.optString("_phone_brain_request")
        val brainDeadline = action.optLong("_phone_brain_deadline", 0L)
        action.remove("_phone_brain_request")
        action.remove("_phone_brain_deadline")
        val kind = action.optString("kind", action.optString("action"))
        action.put("kind", kind)
        fun complete(receipt: JSONObject) { activeActions.remove(request.id); PhoneActionCoordinator.complete(this, request.id, receipt) }
        fun valid() = !isDestroyed && PhoneActionCoordinator.current(request.id) &&
            (brainRequest.isBlank() || (PhoneBrain.isRequestActive(brainRequest) &&
                android.os.SystemClock.elapsedRealtime() < brainDeadline))
        fun approve(details: String, execute: () -> Unit) {
            if (!valid()) { complete(JSONObject().put("ok", false).put("detail", "Action expired.")); return }
            val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), dp(12), dp(20), dp(12)) }
            val scroller = ScrollView(this); scroller.addView(text(details, 14f)); box.addView(scroller, LinearLayout.LayoutParams(-1, dp(260)))
            val slide = SlideToConfirmView(this).apply { label = "SLIDE TO AUTHORISE" }
            box.addView(slide, LinearLayout.LayoutParams(-1, dp(64)))
            val dialog = AlertDialog.Builder(this).setTitle("Review phone action").setView(box).setNegativeButton("Cancel") { _, _ -> complete(JSONObject().put("ok", false).put("detail", "You cancelled this action.")) }.create()
            review = dialog
            dialog.setOnCancelListener { complete(JSONObject().put("ok", false).put("detail", "You cancelled this action.")) }
            slide.onConfirmed = { if (valid()) { dialog.dismiss(); review = null; execute() } }
            dialog.show()
        }
        when (kind) {
            "mcp_authorise" -> approve("SERVER: ${action.optString("server")}\nTOOL: ${action.optString("tool")}\n\n${action.optJSONObject("arguments")?.toString(2)}") {
                complete(JSONObject().put("ok", true).put("authorised", true).put("detail", "You approved this exact MCP call."))
            }
            "capture_dom" -> approve("Read the visible phone screen through Accessibility. Passwords and private authentication fields are excluded.") { OrionAccessibilityService.requestSnapshot(::complete) }
            "click", "scroll", "set_text" -> approve("ACTION: $kind\n\n${action.toString(2)}") {
                val snapshot = action.optString("snapshotId"); val node = action.optString("nodeId")
                when (kind) {
                    "click" -> OrionAccessibilityService.clickNode(snapshot, node, ::complete)
                    "scroll" -> OrionAccessibilityService.scrollNode(snapshot, node, action.optString("direction"), ::complete)
                    else -> OrionAccessibilityService.setNodeText(snapshot, node, action.optString("text"), ::complete)
                }
            }
            "back", "home" -> approve("Go ${kind.uppercase()} on your phone.") { OrionAccessibilityService.navigate(kind, ::complete) }
            "latest_frame" -> complete(JSONObject().put("ok", false).put("detail", "Use screen sharing, then ask what is on the screen. Images are handled directly by the phone AI."))
            else -> IntentRouter(this).route(action, ::approve, ::permission, ::complete, ::valid,
                allowHandsFreeLaunch = true)
        }
    }
    private fun provision() {
        clearError()
        state("Setting up voice", "Connecting to your paired PC once to set up your phone securely…")
        PhoneAiProfile.provision(this) { error -> if (!isDestroyed) {
            if (error == null) state("Voice", "Set-up complete. ORION can answer while the PC is off.")
            else { showError(error); add("meta", error) }
        } }
    }
    private fun showFirstRunSetup() {
        AlertDialog.Builder(this).setTitle("Set up ORION on this phone")
            .setMessage("Your own Gemini API key enables phone replies, microphone transcription and Charon voice. Private PC pairing can copy the PC owner's key to this phone; that copy remains usable until the provider key is rotated. Customer devices should use their own keys.")
            .setItems(arrayOf("Use my own Gemini key", "Pair with my own PC")) { _, choice ->
                if (choice == 0) enterKey() else provision()
            }.setNegativeButton("Later", null).show()
    }
    private fun showPhoneControls() {
        PhoneControls.show(this) { action -> PhoneActionCoordinator.request(this, action) { result ->
            main.post { if (!isDestroyed) addReceipt(action, result) }
        } }
    }
    private fun saveScreenshot() {
        if (captureBusy) { state("Screenshot in progress", "Finish saving the current image first."); return }
        val view = window.decorView
        if (view.width <= 0 || view.height <= 0) {
            state("Screenshot unavailable", "Reopen this screen and try again.")
            return
        }
        captureBusy = true
        main.postDelayed({
            if (isDestroyed || !resumed) { captureBusy = false; return@postDelayed }
            val bitmap = try { Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888) }
                catch (_: OutOfMemoryError) {
                    captureBusy = false
                    state("Screenshot unavailable", "The phone needs more free memory.")
                    return@postDelayed
                }
            try {
                PixelCopy.request(window, bitmap, { result ->
                    if (isDestroyed) {
                        bitmap.recycle(); captureBusy = false
                        return@request
                    }
                    if (result != PixelCopy.SUCCESS) {
                        bitmap.recycle(); captureBusy = false
                        state("Screenshot failed", "Please try again after the screen finishes drawing.")
                        return@request
                    }
                    captureWorker.execute {
                        val file = File(cacheDir, "orion-screenshot-${System.currentTimeMillis()}.png")
                        val prepared = runCatching {
                            FileOutputStream(file).use { output ->
                                if (!bitmap.compress(Bitmap.CompressFormat.PNG, 100, output))
                                    throw IOException("The screenshot could not be encoded.")
                            }
                        }.isSuccess
                        bitmap.recycle()
                        main.post {
                            if (!prepared || isDestroyed) {
                                file.delete(); captureBusy = false
                                if (!isDestroyed) state("Screenshot failed", "Please try again.")
                            } else {
                                pendingScreenshotPath = file.absolutePath
                                runCatching { screenshotDocument.launch(file.name) }.onFailure {
                                    pendingScreenshotPath = null
                                    file.delete(); captureBusy = false
                                    state("Screenshot failed", "The phone could not open a file location.")
                                }
                            }
                        }
                    }
                }, main)
            } catch (_: IllegalArgumentException) {
                bitmap.recycle(); captureBusy = false
                state("Screenshot failed", "Please try again after the screen finishes drawing.")
            }
        }, 250L)
    }
    private fun options() {
        showOptionsSheet("OPTIONS", back = false) { body, sheet ->
            body.addView(text(connectionSummary(), 12f).apply {
                setTextColor(Color.rgb(174, 174, 183))
            }, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(14) })
            body.addView(text("CONTROLS", 10f).apply {
                typeface = Typeface.DEFAULT_BOLD
                letterSpacing = 0.14f
                setTextColor(Color.rgb(181, 181, 188))
            }, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(8) })
            PhoneOptionsMenu.groups.forEach { group ->
                addOptionRow(body, group.title, group.hint, group.marker) {
                    sheet.setOnDismissListener { if (!isFinishing && !isDestroyed) optionGroup(group) }
                    sheet.dismiss()
                }
            }
        }
    }
    /** One honest line: where the face and the answers come from right now. */
    private fun connectionSummary(): String {
        val pc = when {
            pcLink?.connected == true -> "PC linked over ${pcLink?.route ?: "the network"}: the face follows ORION on your PC."
            Prefs.credentials(this) != null -> "PC not reachable: the face follows this phone."
            else -> "No PC paired: the face follows this phone."
        }
        val net = if (PhoneNetwork.online(this)) "Internet available." else
            "Offline: replies wait until the connection returns; time, date and battery still answer."
        val waiting = offlineQueue.size.takeIf { it > 0 }?.let { " $it request(s) waiting." } ?: ""
        return "$pc $net$waiting"
    }
    private fun showOptionsSheet(title: String, back: Boolean,
                                 contents: (LinearLayout, BottomSheetDialog) -> Unit) {
        val sheet = BottomSheetDialog(this)
        val body = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(18), dp(10), dp(18), dp(25))
            background = rounded(Color.rgb(17, 17, 20), dp(24), Color.rgb(59, 43, 48))
        }
        body.addView(View(this).apply { background = rounded(Color.rgb(100, 100, 108), dp(3)) },
            LinearLayout.LayoutParams(dp(38), dp(4)).apply {
                gravity = Gravity.CENTER_HORIZONTAL
                bottomMargin = dp(17)
            })
        val heading = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        if (back) heading.addView(text("‹", 30f).apply {
            gravity = Gravity.CENTER
            contentDescription = "Back to options"
            isClickable = true
            isFocusable = true
            setOnClickListener {
                sheet.setOnDismissListener { if (!isFinishing && !isDestroyed) options() }
                sheet.dismiss()
            }
        }, LinearLayout.LayoutParams(dp(42), dp(46)).apply { rightMargin = dp(6) })
        heading.addView(text(title, 19f).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.BOLD)
            letterSpacing = 0.09f
            setTextColor(Color.WHITE)
        }, LinearLayout.LayoutParams(0, -2, 1f))
        heading.addView(text("×", 26f).apply {
            gravity = Gravity.CENTER
            contentDescription = "Close options"
            isClickable = true
            isFocusable = true
            setOnClickListener { sheet.dismiss() }
        }, LinearLayout.LayoutParams(dp(42), dp(46)))
        body.addView(heading, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(18) })
        contents(body, sheet)
        val viewport = ScrollView(this).apply {
            isFillViewport = false
            clipToPadding = false
            addView(body, ViewGroup.LayoutParams(-1, -2))
        }
        sheet.setContentView(viewport)
        sheet.setOnShowListener {
            sheet.findViewById<View>(com.google.android.material.R.id.design_bottom_sheet)
                ?.setBackgroundColor(Color.TRANSPARENT)
            sheet.window?.setDimAmount(0.68f)
        }
        sheet.show()
    }
    private fun addOptionRow(parent: LinearLayout, label: String, detail: String = "",
                             marker: String = "", action: () -> Unit) {
        val row = LinearLayout(this).apply {
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(14), dp(10), dp(13), dp(10))
            background = rounded(Color.rgb(27, 27, 31), dp(14), Color.rgb(52, 52, 59))
            isClickable = true
            isFocusable = true
            contentDescription = if (detail.isBlank()) label else "$label, $detail"
            setOnClickListener { action() }
        }
        if (marker.isNotBlank()) row.addView(text(marker, 15f).apply {
            gravity = Gravity.CENTER
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.rgb(236, 104, 115))
            background = rounded(Color.rgb(62, 27, 35), dp(10))
        }, LinearLayout.LayoutParams(dp(38), dp(38)).apply { rightMargin = dp(12) })
        val caption = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        caption.addView(text(label, 15f).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(Color.WHITE)
        })
        if (detail.isNotBlank()) caption.addView(text(detail, 11f).apply {
            setTextColor(Color.rgb(159, 159, 168))
        })
        row.addView(caption, LinearLayout.LayoutParams(0, -2, 1f))
        row.addView(text("›", 22f).apply {
            gravity = Gravity.CENTER
            setTextColor(Color.rgb(176, 94, 103))
        }, LinearLayout.LayoutParams(dp(24), -2))
        parent.addView(row, LinearLayout.LayoutParams(-1, dp(if (detail.isBlank()) 57 else 67)).apply {
            bottomMargin = dp(8)
        })
    }
    private fun optionGroup(group: PhoneOptionGroup) {
        showOptionsSheet(group.title.uppercase(), back = true) { body, sheet ->
            group.actions.forEach { index ->
                val (label, detail) = optionLabel(index)
                addOptionRow(body, label, detail) {
                    sheet.setOnDismissListener { if (!isFinishing && !isDestroyed) option(index) }
                    sheet.dismiss()
                }
            }
        }
    }
    private fun optionLabel(index: Int): Pair<String, String> = when (index) {
        0 -> "Phone controls" to "Apps, calls and messages"
        1 -> "Background listening" to if (PhoneListeningService.running) "On" else "Off"
        2 -> "Stop speaking" to ""
        3 -> "Screen sharing" to if (OrionProjectionService.isCapturing) "On" else "Off"
        4 -> "Accessibility" to "Phone controls permission"
        5 -> "Sentinel" to if (OrionSentinelService.running) "On" else "Off"
        6 -> "Connect MCP" to ""
        7 -> "Disconnect MCP" to ""
        8 -> "Provision from PC" to ""
        9 -> "Provisioning address" to ""
        10 -> "Gemini key" to ""
        11 -> "Spoken replies" to if (getPreferences(MODE_PRIVATE).getBoolean("voice", true)) "On" else "Off"
        12 -> "Clear conversation" to ""
        13 -> "Desktop connection" to ""
        14 -> "Remove Gemini key" to ""
        15 -> "Offline UK map" to ""
        16 -> "Voice engine" to if (PhoneAiProfile.liveVoiceEnabled(this)) "Live" else "Direct"
        17 -> "Text AI provider" to ""
        18 -> "Remove other AI keys" to ""
        19 -> "Performance" to "Response times"
        20 -> "Gemini text model" to ""
        21 -> "Capabilities" to "What ORION can do"
        22 -> "Require wake word" to if (PhoneListeningService.requiresWake(this)) "On" else "Off"
        23 -> "Save screenshot" to "Choose a file on this phone"
        24 -> "Voice" to PhoneAiProfile.voiceName(this)
        25 -> "PC controls" to "Shut down your paired PC"
        else -> "Option" to ""
    }
    private fun option(which: Int) {
            clearError()
            state(if (PhoneListeningService.running) "Listening" else "Voice", "")
            when (which) {
            0 -> showPhoneControls()
            1 -> if (PhoneListeningService.running) {
                PhoneListeningService.stop(this)
                state("Voice", "Hands-free listening is off.")
            } else startListening()
            2 -> { PhoneVoice.stopAll(); RemoteSpeech.stopAll(); state("Voice", "Speech stopped.") }
            3 -> screenSharing()
            4 -> startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            5 -> if (OrionSentinelService.running) OrionSentinelService.stop(this) else {
                if (Settings.canDrawOverlays(this)) OrionSentinelService.start(this)
                else startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:$packageName")))
            }
            6 -> configureMcp()
            7 -> { PhoneMcpClient.clear(this); state("MCP disconnected", "Your phone tools remain available.") }
            8 -> provision()
            9 -> changeProvisioningAddress()
            10 -> enterKey()
            11 -> { val prefs = getPreferences(MODE_PRIVATE); val enabled = !prefs.getBoolean("voice", true); prefs.edit().putBoolean("voice", enabled).apply(); if (!enabled) PhoneVoice.stopAll(); state("Spoken replies ${if (enabled) "on" else "off"}", "Your preference is saved on the phone.") }
            12 -> AlertDialog.Builder(this).setTitle("Clear this phone's conversation?").setMessage("This removes the encrypted phone history.").setPositiveButton("Clear") { _, _ ->
                cancelTurn()
                historyGeneration.incrementAndGet()
                PhoneConversationStore.clear(this)
                shownHistoryId = ""
                messages.removeAllViews()
                messages.addView(emptyConversation)
                emptyConversation.visibility = View.VISIBLE
            }.setNegativeButton("Cancel", null).show()
            13 -> { PhoneAiProfile.setPhoneMode(this, false); startActivity(Intent(this, MainActivity::class.java).putExtra("desktop_mode", true)); finish() }
            14 -> { cancelTurn(); PhoneListeningService.stop(this); PhoneAiProfile.forget(this)
                state("Gemini key removed", if (PhoneHostedProfile.ready(this))
                    "Text replies and phone speech remain available. Cloud audio needs a Gemini key." else
                    "Provision or enter a Gemini key to resume replies.") }
            15 -> offlineMaps()
            16 -> {
                val enabled = !PhoneAiProfile.liveVoiceEnabled(this)
                PhoneAiProfile.setLiveVoiceEnabled(this, enabled)
                PhoneVoice.stopAll()
                Handler(Looper.getMainLooper()).post { voice.prepare() }
                state("Voice", "")
            }
            17 -> configureTextProvider()
            18 -> AlertDialog.Builder(this).setTitle("Remove other text AI keys?")
                .setMessage("This removes every non-Gemini text AI key from this phone and selects Gemini for replies.")
                .setPositiveButton("Remove") { _, _ ->
                    changeAiConfiguration { PhoneHostedProfile.forget(this) }
                    state("Text AI keys removed", "Gemini is selected for replies.")
                }.setNegativeButton("Cancel", null).show()
            19 -> AlertDialog.Builder(this).setTitle("Phone response times")
                .setMessage(PhonePerformance.summary(this))
                .setPositiveButton("Close", null).show()
            20 -> chooseGeminiModel()
            21 -> AlertDialog.Builder(this).setTitle("What ORION can do")
                .setMessage(PhoneCapabilities.summary(this))
                .setPositiveButton("Phone controls") { _, _ -> showPhoneControls() }
                .setNegativeButton("Close", null).show()
            22 -> {
                val required = !PhoneListeningService.requiresWake(this)
                PhoneListeningService.setRequiresWake(this, required)
                state("Listening", if (required) "Say ‘ORION’ before each request."
                    else "Speak naturally while Background listening is on.")
            }
            23 -> saveScreenshot()
            24 -> chooseVoice()
            25 -> {
                pcControls?.close()
                pcControls = RemotePcControls(this).also { it.show() }
            }
            }
    }

    private fun chooseVoice() {
        showOptionsSheet("VOICE", back = true) { body, sheet ->
            val selected = PhoneAiProfile.voiceName(this)
            fun row(name: String, detail: String) {
                addOptionRow(body, name, detail, if (selected == name) "✓" else "") {
                    if (!PhoneAiProfile.selectVoice(this, name)) {
                        showError("The phone could not save this voice.")
                        return@addOptionRow
                    }
                    cancelTurn()
                    PhoneVoice.stopAll(); RemoteSpeech.stopAll()
                    Handler(Looper.getMainLooper()).post { voice.prepare() }
                    sheet.dismiss()
                    state("Voice", "")
                }
            }
            row(PhoneVoices.PHONE, "Phone speech engine")
            PhoneVoices.cloud.forEach { row(it.name, it.delivery) }
        }
    }

    private fun chooseGeminiModel() {
        fun checkAndSave(selected: String) {
            val key = PhoneAiProfile.key(this)
            if (key == null) { showError("Enter your Gemini key before choosing a model."); return }
            state("Checking model", "")
            PhoneAiProfile.verifyTextModel(this, key, selected) { error ->
                if (!isDestroyed) {
                    if (error != null) showError(error)
                    else changeAiConfiguration {
                        if (PhoneAiProfile.selectTextModel(this, selected))
                            state("Model checked", "$selected is available for text generation with your key.")
                        else showError("The model could not be saved.")
                    }
                }
            }
        }
        val choices = arrayOf("gemini-3.5-flash-lite (default)", "gemini-3.5-flash", "Enter another text model ID")
        AlertDialog.Builder(this).setTitle("Gemini text model")
            .setMessage("ORION will use the model you choose for phone replies. It will not silently switch to a larger model. Check your own key's model access and usage limits.")
            .setItems(choices) { _, index ->
                if (index < 2) {
                    val selected = choices[index].substringBefore(' ')
                    checkAndSave(selected)
                } else {
                    val entry = EditText(this).apply {
                        setSingleLine(true)
                        hint = "gemini-text-model-id"
                        setText(PhoneAiProfile.model(this@PhoneConsoleActivity))
                    }
                    AlertDialog.Builder(this).setTitle("Text model ID")
                        .setMessage("Use a Gemini text model that supports function calls. Voice and transcription use the separate Gemini audio path.")
                        .setView(entry).setPositiveButton("Save") { _, _ ->
                            val selected = entry.text.toString().trim()
                            checkAndSave(selected)
                        }.setNegativeButton("Cancel", null).show()
                }
            }.show()
    }

    private fun configureTextProvider() {
        val providers = HostedTextProviders.choices
        val labels = providers.map(HostedTextProviders::label).toTypedArray()
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(4), dp(20), dp(4))
        }
        val picker = Spinner(this).apply {
            adapter = ArrayAdapter(this@PhoneConsoleActivity, android.R.layout.simple_spinner_dropdown_item, labels)
            setSelection(providers.indexOf(PhoneHostedProfile.selected(this@PhoneConsoleActivity)).coerceAtLeast(0))
        }
        val key = EditText(this).apply {
            hint = "API key"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
            setSingleLine(true)
        }
        val model = EditText(this).apply {
            hint = "Chat model ID"
            inputType = InputType.TYPE_CLASS_TEXT
            setSingleLine(true)
        }
        val account = EditText(this).apply {
            hint = "Cloudflare account ID"
            inputType = InputType.TYPE_CLASS_TEXT
            setSingleLine(true)
            setText(PhoneHostedProfile.accountId(this@PhoneConsoleActivity))
        }
        box.addView(picker); box.addView(key); box.addView(model); box.addView(account)
        picker.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) {}
            override fun onItemSelected(parent: android.widget.AdapterView<*>?, view: View?, position: Int, id: Long) {
                val provider = providers[position]
                // A key typed for one service must never follow a spinner change.
                key.text.clear()
                val gemini = provider == HostedTextProviders.GEMINI
                key.visibility = if (gemini) View.GONE else View.VISIBLE
                model.visibility = if (gemini) View.GONE else View.VISIBLE
                account.visibility = if (provider == HostedTextProviders.CLOUDFLARE) View.VISIBLE else View.GONE
                if (!gemini) {
                    model.setText(PhoneHostedProfile.model(this@PhoneConsoleActivity, provider))
                    key.hint = if (PhoneHostedProfile.hasKey(this@PhoneConsoleActivity, provider))
                        "Key stored; leave blank to keep it" else "API key"
                }
            }
        }
        AlertDialog.Builder(this).setTitle("Text AI provider")
            .setMessage("Phone replies and reviewed actions use this provider. Microphone transcription and cloud speech use the Gemini key; phone speech needs no API key. Keys stay encrypted on the phone. Check a model's price before selecting a paid ID.")
            .setView(box).setPositiveButton("Save") { _, _ ->
                changeAiConfiguration {
                    val provider = providers[picker.selectedItemPosition.coerceIn(providers.indices)]
                    if (provider == HostedTextProviders.GEMINI) {
                        PhoneHostedProfile.selectGemini(this)
                        state("Gemini selected", if (PhoneAiProfile.hasKey(this)) "Ready for replies." else "Enter a Gemini key to continue.")
                    } else {
                        val saved = PhoneHostedProfile.save(this, provider, key.text.toString().trim(),
                            model.text.toString().trim(), account.text.toString().trim())
                        state(if (saved) "${labels[picker.selectedItemPosition]} selected" else "Text AI set-up failed",
                            if (saved) "Ready for phone replies and reviewed actions." else
                                "Check the provider key, model ID and Cloudflare account ID if used.")
                    }
                }
            }.setNegativeButton("Cancel", null).show()
    }
    private fun offlineMaps() {
        startActivity(Intent(this, OfflineMapActivity::class.java))
    }
    private fun changeProvisioningAddress() {
        val value = EditText(this).apply {
            hint = "https://your-host.trycloudflare.com"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setText(Prefs.baseUrl(this@PhoneConsoleActivity).orEmpty())
        }
        AlertDialog.Builder(this).setTitle("PC address for one-time setup")
            .setMessage("Enter the current HTTPS tunnel or Tailscale Serve address. The paired PC connection supports approved key setup, PC controls and Desktop mode. Normal phone replies remain independent.")
            .setView(value).setPositiveButton("Save and provision") { _, _ ->
                val raw = value.text.toString().trim()
                val parsed = Prefs.parseAddress(raw)
                if (!raw.startsWith("https://", ignoreCase = true) || parsed == null || parsed.https != true ||
                    raw.contains('@') || parsed.host.matches(Regex("100\\.\\d+\\.\\d+\\.\\d+"))) {
                    state("Invalid HTTPS address", "Enter the full https:// address shown by the PC tunnel.")
                } else {
                    val port = parsed.port ?: 443
                    getSharedPreferences(Prefs.NAME, MODE_PRIVATE).edit().putString(Prefs.HOST, parsed.host)
                        .putInt(Prefs.PORT, port).apply()
                    Prefs.clearEndpoints(this)
                    provision()
                }
            }.setNegativeButton("Cancel", null).show()
    }
    private fun screenSharing() {
        if (OrionProjectionService.isCapturing) { OrionProjectionService.stop(this); state("Screen sharing off", "The cached screen image has been cleared.") }
        else projection.launch((getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager).createScreenCaptureIntent())
    }
    private fun enterKey() {
        val value = EditText(this).apply { hint = "Gemini API key"; inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD }
        AlertDialog.Builder(this).setTitle("Phone AI key").setMessage("Stored encrypted in Android Keystore. Sent only to Google's Gemini API.").setView(value)
            .setPositiveButton("Save") { _, _ ->
                val key = value.text.toString().trim()
                val model = PhoneAiProfile.model(this)
                state("Checking key", "")
                PhoneAiProfile.verifyTextModel(this, key, model) { error ->
                    if (!isDestroyed) {
                        if (error != null) showError(error)
                        else changeAiConfiguration {
                            if (PhoneAiProfile.save(this, key, model, PhoneAiProfile.voiceModel(this)))
                                state("Key and model checked", "The key is sealed on this phone. Charon voice still depends on its separate audio model and quota.")
                            else showError("Android Keystore could not seal the key; the previous key was kept.")
                        }
                    }
                }
            }.setNegativeButton("Cancel", null).show()
    }
    private fun configureMcp() {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(20), 0, dp(20), 0) }
        val url = EditText(this).apply { hint = "HTTPS MCP endpoint"; inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI }
        val token = EditText(this).apply { hint = "Bearer token, if required"; inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD }
        box.addView(url); box.addView(token)
        AlertDialog.Builder(this).setTitle("Connect phone MCP").setMessage("Connect a service you control. Its tools will require native review before each call.").setView(box)
            .setPositiveButton("Connect") { _, _ -> if (PhoneMcpClient.save(this, url.text.toString().trim(), token.text.toString())) {
                worker.execute { val client = PhoneMcpClient(this); val found = runCatching { client.discover() }.getOrNull(); client.close(); main.post { if (!isDestroyed) state(if (found != null) "MCP connected" else "MCP connection failed", if (found != null) "${found.length()} tools discovered on the phone." else "Check the HTTPS endpoint and its access token.") } }
            } else state("Invalid MCP configuration", "Use an HTTPS endpoint without a credential in its address.") }.setNegativeButton("Cancel", null).show()
    }
    private fun handleSentinel(source: Intent?) {
        when (source?.getStringExtra(SentinelActionContract.EXTRA_ACTION)) {
            SentinelActionContract.DICTATE -> main.postDelayed({ dictate() }, 250)
            SentinelActionContract.TOGGLE_LISTENING -> main.postDelayed({ if (PhoneListeningService.running) PhoneListeningService.stop(this) else startListening() }, 250)
            SentinelActionContract.SCREEN_SHARING -> main.postDelayed({ screenSharing() }, 250)
            SentinelActionContract.PHONE_CONTROLS -> main.postDelayed({ showPhoneControls() }, 250)
        }
        source?.removeExtra(SentinelActionContract.EXTRA_ACTION)
    }
    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); setIntent(intent); handleSentinel(intent) }
    override fun onResume() {
        super.onResume()
        resumed = true
        if (::avatar.isInitialized) avatar.resumeVisual()
        main.removeCallbacks(meter)
        main.post(meter)
        PhoneActionCoordinator.attach(this, actionHandler)
        if (::messages.isInitialized) loadHistory()
        // Follow the PC's face while it is reachable (home Wi-Fi first). Only
        // a paired phone tries; with no pairing nothing leaves the phone.
        if (pcLink == null && Prefs.credentials(this) != null) pcLink = PcFaceLink(this,
            onEmotion = { name, intensity -> main.post { if (!isDestroyed) avatar.emotion(name, intensity) } },
            onStyle = { style -> main.post { if (!isDestroyed) avatar.style(style) } }).also { it.start() }
    }
    override fun onPause() {
        resumed = false
        pcLink?.stop(); pcLink = null
        if (::avatar.isInitialized) { avatar.link(false); avatar.pauseVisual() }
        main.removeCallbacks(meter)
        PhoneActionCoordinator.detach(actionHandler)
        super.onPause()
    }
    override fun onDestroy() {
        pcControls?.close()
        generation.incrementAndGet(); historyGeneration.incrementAndGet()
        main.removeCallbacksAndMessages(null); review?.dismiss()
        if (::dictation.isInitialized) dictation.close()
        if (::brain.isInitialized) brain.close()
        if (::voice.isInitialized) voice.close()
        if (::avatar.isInitialized) avatar.dispose()
        if (isFinishing) pendingScreenshotPath?.let { File(it).delete() }
        worker.shutdownNow(); PhoneActionCoordinator.detach(actionHandler)
        captureWorker.shutdown()
        super.onDestroy()
    }
    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("pending_screenshot", pendingScreenshotPath)
        super.onSaveInstanceState(outState)
    }
}
