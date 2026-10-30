package com.orion.companion

import android.Manifest
import android.content.ContentUris
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.CalendarContract
import android.text.InputType
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.KeyEvent
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.GridLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.snackbar.Snackbar
import org.json.JSONObject
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneOffset
import java.time.ZonedDateTime
import java.time.format.TextStyle
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Calendar, Focus and Fitness on the phone - the same three pages as the PC's
 * Command Deck, working with the PC off. The calendar is the phone's own;
 * workouts and focus blocks are kept on the phone, encrypted (DayStore).
 */
class DayActivity : AppCompatActivity() {
    companion object {
        const val EXTRA_TAB = "tab"
        const val CALENDAR = "calendar"
        const val FOCUS = "focus"
        const val FITNESS = "fitness"
        private val TABS = listOf(CALENDAR to "Calendar", FOCUS to "Focus", FITNESS to "Fitness")

        fun open(context: Context, tab: String) =
            context.startActivity(Intent(context, DayActivity::class.java).putExtra(EXTRA_TAB, tab))
    }

    private val calendarPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { if (tab == CALENDAR) show(CALENDAR) }
    private val notificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }
    private val main = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private var doc = JSONObject()
    private var loaded = false
    private var tab = CALENDAR
    private lateinit var content: FrameLayout
    private val tabButtons = mutableMapOf<String, TextView>()

    // Focus page
    private var mode = DayPlanner.MODES[1]
    private var dial: FocusDialView? = null
    private var focusControls: LinearLayout? = null
    private var focusStatus: TextView? = null
    private var company: LinearLayout? = null
    private var face: PhoneAvatarView? = null
    private var stepList: LinearLayout? = null
    private var focusStats: List<TextView> = emptyList()
    private var lastControls = ""
    private val ticker = object : Runnable {
        override fun run() {
            if (tab == FOCUS && !isFinishing) { drawDial(); main.postDelayed(this, 1000) }
        }
    }

    // Fitness page
    private var kind = "run"
    private var minutes = 30
    private var distance = 0.0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        tab = intent.getStringExtra(EXTRA_TAB)?.takeIf { t -> TABS.any { it.first == t } } ?: CALENDAR
        val page = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(8), dp(16), 0)
            setBackgroundColor(Color.rgb(8, 8, 10))
        }
        val header = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        header.addView(text("‹", 30f).apply {
            gravity = Gravity.CENTER
            contentDescription = "Back"
            isClickable = true
            isFocusable = true
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(dp(42), dp(52)))
        val segments = LinearLayout(this).apply {
            setPadding(dp(4), dp(4), dp(4), dp(4))
            background = rounded(Color.rgb(20, 20, 23), dp(16), Color.rgb(44, 44, 50))
        }
        TABS.forEach { (key, name) ->
            val segment = text(name, 14f).apply {
                gravity = Gravity.CENTER
                isClickable = true
                isFocusable = true
                setOnClickListener { show(key) }
            }
            tabButtons[key] = segment
            segments.addView(segment, LinearLayout.LayoutParams(0, dp(38), 1f))
        }
        header.addView(segments, LinearLayout.LayoutParams(0, -2, 1f).apply { rightMargin = dp(42) })
        page.addView(header, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        content = FrameLayout(this)
        page.addView(content, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(page)
        show(tab)
        io.execute {
            val read = DayStore.read(this)
            main.post {
                if (isDestroyed) return@post
                doc = read
                loaded = true
                mode = DayPlanner.MODES.firstOrNull { it.key == doc.optString("mode") } ?: mode
                show(tab)
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        intent.getStringExtra(EXTRA_TAB)?.let { show(it) }
    }

    override fun onResume() {
        super.onResume()
        face?.resumeVisual()
        if (tab == FOCUS) { main.removeCallbacks(ticker); main.post(ticker) }
    }

    override fun onPause() {
        main.removeCallbacks(ticker)
        face?.pauseVisual()
        super.onPause()
    }

    override fun onDestroy() {
        main.removeCallbacksAndMessages(null)
        face?.dispose()
        face = null
        io.shutdown()
        super.onDestroy()
    }

    private fun show(key: String) {
        tab = key
        tabButtons.forEach { (k, view) ->
            val on = k == key
            view.setTextColor(if (on) Color.WHITE else Color.rgb(150, 150, 158))
            view.typeface = Typeface.create("sans-serif-medium", if (on) Typeface.BOLD else Typeface.NORMAL)
            view.background = if (on) rounded(Color.rgb(48, 26, 31), dp(12), Color.rgb(110, 40, 48)) else null
        }
        main.removeCallbacks(ticker)
        face?.dispose()
        face = null
        dial = null
        content.removeAllViews()
        val body = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 0, 0, dp(24)) }
        content.addView(ScrollView(this).apply { isFillViewport = true; addView(body) })
        when (key) {
            CALENDAR -> buildCalendar(body)
            FOCUS -> { buildFocus(body); main.post(ticker) }
            else -> buildFitness(body)
        }
    }

    // ── calendar ───────────────────────────────────────────────────────────

    private fun buildCalendar(body: LinearLayout) {
        val next = card(body)
        next.addView(caption("Next up"))
        val nextTitle = text("", 22f).apply {
            setTextColor(Color.WHITE)
            typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
        }
        val nextMeta = text("", 13f).apply { setTextColor(Color.rgb(236, 104, 115)) }
        next.addView(nextTitle)
        next.addView(nextMeta)

        val add = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        val field = input("Add to your calendar…")
        val addButton = button("Add", accent = true) { addEvent(field) }
        field.setOnEditorActionListener { _, action, event ->
            val done = action == EditorInfo.IME_ACTION_DONE ||
                (event?.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN)
            if (done) addEvent(field)
            done
        }
        add.addView(field, LinearLayout.LayoutParams(0, dp(48), 1f).apply { rightMargin = dp(8) })
        add.addView(addButton, LinearLayout.LayoutParams(dp(72), dp(48)))
        body.addView(add, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(12) })

        val week = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        body.addView(week)
        if (checkSelfPermission(Manifest.permission.READ_CALENDAR) != PackageManager.PERMISSION_GRANTED) {
            nextTitle.text = "Your phone calendar"
            nextMeta.text = ""
            val ask = card(week)
            ask.addView(text("ORION reads the calendars on this phone to show your week. Nothing leaves the phone.", 14f))
            ask.addView(button("Show my calendar", accent = true) {
                calendarPermission.launch(Manifest.permission.READ_CALENDAR)
            }, LinearLayout.LayoutParams(-1, dp(48)).apply { topMargin = dp(10) })
            return
        }
        nextTitle.text = "…"
        io.execute {
            val now = ZonedDateTime.now()
            val entries = runCatching { readCalendar(now) }
            main.post {
                if (isDestroyed || tab != CALENDAR) return@post
                val found = entries.getOrDefault(emptyList())
                val first = DayPlanner.nextUp(found, now)
                nextTitle.text = first?.title ?: "Nothing coming up"
                nextMeta.text = first?.let {
                    listOfNotNull(DayPlanner.relative(it, now), it.at?.let { _ -> it.timeText() }, it.source)
                        .joinToString("  ·  ")
                } ?: if (entries.isFailure) "The calendar could not be read." else "The week is clear."
                drawWeek(week, DayPlanner.weekOf(found, now.toLocalDate()), now.toLocalDate())
            }
        }
    }

    private fun readCalendar(now: ZonedDateTime): List<DayPlanner.Entry> {
        val zone = now.zone
        val start = now.toLocalDate().atStartOfDay(zone)
        val end = start.plusDays(7)
        val uri = CalendarContract.Instances.CONTENT_URI.buildUpon().also {
            ContentUris.appendId(it, start.toInstant().toEpochMilli())
            ContentUris.appendId(it, end.toInstant().toEpochMilli())
        }.build()
        val columns = arrayOf(CalendarContract.Instances.TITLE, CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.END, CalendarContract.Instances.ALL_DAY,
            CalendarContract.Instances.CALENDAR_DISPLAY_NAME)
        val out = mutableListOf<DayPlanner.Entry>()
        contentResolver.query(uri, columns, null, null, CalendarContract.Instances.BEGIN + " ASC")?.use { c ->
            while (c.moveToNext() && out.size < 200) {
                val title = c.getString(0)?.trim().orEmpty().ifEmpty { "Busy" }
                val begin = c.getLong(1)
                val finish = c.getLong(2)
                val source = c.getString(4)?.trim().orEmpty().ifEmpty { "Calendar" }
                if (c.getInt(3) == 1) {
                    // All-day events are stored as UTC midnights; the end is exclusive.
                    var day = Instant.ofEpochMilli(begin).atZone(ZoneOffset.UTC).toLocalDate()
                    val last = Instant.ofEpochMilli(maxOf(begin, finish - 1)).atZone(ZoneOffset.UTC).toLocalDate()
                    while (!day.isAfter(last) && day.isBefore(end.toLocalDate())) {
                        if (!day.isBefore(start.toLocalDate())) out += DayPlanner.Entry(day, title, null, source)
                        day = day.plusDays(1)
                    }
                } else {
                    val at = Instant.ofEpochMilli(begin).atZone(zone)
                    out += DayPlanner.Entry(at.toLocalDate(), title, at, source)
                }
            }
        }
        return out
    }

    private fun drawWeek(week: LinearLayout, days: List<Pair<LocalDate, List<DayPlanner.Entry>>>, today: LocalDate) {
        week.removeAllViews()
        days.forEach { (day, entries) ->
            val box = card(week, tight = true)
            val heading = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
            heading.addView(text(if (day == today) "Today" else day.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.UK), 15f).apply {
                setTextColor(if (day == today) Color.WHITE else Color.rgb(200, 200, 206))
                typeface = Typeface.create("sans-serif-medium", if (day == today) Typeface.BOLD else Typeface.NORMAL)
            }, LinearLayout.LayoutParams(0, -2, 1f))
            heading.addView(text("${day.dayOfMonth} ${day.month.getDisplayName(TextStyle.SHORT, Locale.UK)}", 12f).apply {
                setTextColor(Color.rgb(130, 130, 138))
            })
            box.addView(heading)
            if (entries.isEmpty()) box.addView(text("Free", 13f).apply { setTextColor(Color.rgb(110, 110, 118)) })
            entries.forEach { entry ->
                val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; setPadding(0, dp(4), 0, dp(4)) }
                row.addView(View(this).apply { background = rounded(Color.rgb(211, 47, 47), dp(2)) },
                    LinearLayout.LayoutParams(dp(3), dp(30)).apply { rightMargin = dp(10) })
                row.addView(text(entry.timeText(), 12f).apply {
                    typeface = Typeface.MONOSPACE
                    setTextColor(Color.rgb(176, 176, 184))
                }, LinearLayout.LayoutParams(dp(64), -2))
                row.addView(text(entry.title, 15f).apply { setTextColor(Color.rgb(242, 242, 244)) },
                    LinearLayout.LayoutParams(0, -2, 1f))
                row.contentDescription = "${entry.timeText()}, ${entry.title}, ${entry.source}"
                box.addView(row)
            }
        }
    }

    private fun addEvent(field: EditText) {
        val title = field.text.toString().trim()
        if (title.isEmpty()) return
        val start = ZonedDateTime.now().plusHours(1).withMinute(0).withSecond(0).withNano(0)
        runCatching {
            startActivity(Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI)
                .putExtra(CalendarContract.Events.TITLE, title)
                .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, start.toInstant().toEpochMilli())
                .putExtra(CalendarContract.EXTRA_EVENT_END_TIME, start.plusHours(1).toInstant().toEpochMilli()))
            field.setText("")
        }.onFailure { say("This phone has no calendar app to add it to.") }
    }

    // ── focus ──────────────────────────────────────────────────────────────

    private fun block(): DayPlanner.Block? = DayPlanner.Block.fromJson(doc.optJSONObject("block"))
    private fun steps(): List<DayPlanner.Step> = DayPlanner.stepsFromJson(doc.optJSONArray("steps"))
    private fun focusCaption(): String = steps().firstOrNull { !it.done }?.text ?: doc.optString("task")

    private fun buildFocus(body: LinearLayout) {
        val chips = LinearLayout(this).apply { gravity = Gravity.CENTER }
        DayPlanner.MODES.forEach { m ->
            chips.addView(chip(m.name, m == mode) {
                if (block() == null) { mode = m; save({ it.put("mode", m.key) }) { show(FOCUS) } }
            }, LinearLayout.LayoutParams(-2, dp(36)).apply { rightMargin = dp(6) })
        }
        body.addView(HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            addView(chips)
        }, LinearLayout.LayoutParams(-2, -2).apply { gravity = Gravity.CENTER_HORIZONTAL })

        dial = FocusDialView(this)
        body.addView(dial, LinearLayout.LayoutParams(-1, dp(300)))

        company = LinearLayout(this).apply { gravity = Gravity.CENTER; visibility = View.GONE }
        body.addView(company, LinearLayout.LayoutParams(-1, -2))

        focusControls = LinearLayout(this).apply { gravity = Gravity.CENTER }
        body.addView(focusControls, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(6) })
        focusStatus = text("", 13f).apply {
            gravity = Gravity.CENTER
            setTextColor(Color.rgb(160, 160, 168))
        }
        body.addView(focusStatus, LinearLayout.LayoutParams(-1, -2).apply { topMargin = dp(4); bottomMargin = dp(10) })

        val plan = card(body)
        val task = input("What are you working on?").apply {
            setText(doc.optString("task"))
            imeOptions = EditorInfo.IME_ACTION_DONE
            setOnFocusChangeListener { _, has -> if (!has) saveTask(this) }
            setOnEditorActionListener { _, _, _ -> saveTask(this); false }
        }
        plan.addView(task, LinearLayout.LayoutParams(-1, dp(48)))
        stepList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(6), 0, dp(6)) }
        plan.addView(stepList)
        val step = input("Add a step").apply { imeOptions = EditorInfo.IME_ACTION_DONE }
        step.setOnEditorActionListener { _, _, _ -> addSteps(step); true }
        plan.addView(step, LinearLayout.LayoutParams(-1, dp(48)))

        val stats = card(body)
        val row = LinearLayout(this)
        focusStats = listOf("Today", "Blocks", "Day streak").map { name ->
            val column = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
            val value = text("0", 24f).apply {
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
            }
            column.addView(value)
            column.addView(text(name, 11f).apply { setTextColor(Color.rgb(140, 140, 148)) })
            row.addView(column, LinearLayout.LayoutParams(0, -2, 1f))
            value
        }
        stats.addView(row)
        drawSteps()
        drawFocusStats()
        lastControls = ""
        drawDial()
    }

    private fun drawDial() {
        val view = dial ?: return
        val now = System.currentTimeMillis()
        val block = block()
        var breakUntil = doc.optLong("break_until")
        if (block == null && breakUntil in 1..now) breakUntil = 0
        val state = DayPlanner.dial(block, now, mode.minutes, breakUntil, doc.optLong("break_ms"), focusCaption())
        view.dial = state
        val companyOn = block?.bodyDouble == true && state.mode in setOf("running", "due")
        showCompany(companyOn)
        if (companyOn) window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (state.mode != lastControls) { lastControls = state.mode; drawControls(state.mode) }
    }

    private fun drawControls(state: String) {
        val row = focusControls ?: return
        row.removeAllViews()
        val buttons = when (state) {
            "running" -> listOf(button("Done", accent = true) { finishBlock(kept = true) },
                button("Distracted") { distracted() }, button("Stop") { stopBlock() })
            "due" -> listOf(button("Take a break", accent = true) { takeBreak() }, button("Stop") { finishBlock(kept = true) })
            "break" -> listOf(button("Start", accent = true) { startBlock() }.apply { text = "Back to it" })
            else -> listOf(button("Start", accent = true) { startBlock() })
        }
        buttons.forEach { row.addView(it, LinearLayout.LayoutParams(0, dp(50), 1f).apply { marginStart = dp(4); marginEnd = dp(4) }) }
    }

    private fun showCompany(on: Boolean) {
        val row = company ?: return
        if (on && face == null) {
            face = PhoneAvatarView(this).apply { contentDescription = "ORION, keeping you company" }
            row.addView(face, LinearLayout.LayoutParams(dp(96), dp(100)).apply { rightMargin = dp(10) })
            row.addView(text("I'm here. I'll check in halfway.", 13f).apply { setTextColor(Color.rgb(176, 176, 184)) })
        }
        row.visibility = if (on) View.VISIBLE else View.GONE
    }

    private fun startBlock() {
        if (!loaded) return
        askForNotifications()
        val now = System.currentTimeMillis()
        val label = focusCaption().ifBlank { "Focus" }
        val next = DayPlanner.Block(label, now, mode.minutes, DayPlanner.breakFor(mode.minutes), mode.bodyDouble)
        FocusNudges.cancelAll(this)
        FocusNudges.schedule(this, "break", "Time for a short break", now + next.totalMs())
        if (next.bodyDouble && next.minutes >= 10)
            FocusNudges.schedule(this, "halfway", "Halfway there. Still with you.", now + next.totalMs() / 2)
        confirmTap(dial)
        save({ it.put("block", next.toJson()); it.remove("break_until"); it.remove("break_ms") }) {
            focusStatus?.text = ""
            drawDial()
        }
    }

    private fun finishBlock(kept: Boolean) {
        val block = block() ?: return
        val now = System.currentTimeMillis()
        val minutes = minOf(block.minutes, Math.round(block.elapsedMs(now) / 60_000.0).toInt())
        FocusNudges.cancelAll(this)
        save({ doc ->
            if (kept) DayStore.recordFocus(doc, now, maxOf(1, minutes))
            doc.remove("block")
        }) {
            focusStatus?.text = if (kept) "$minutes min done." else "Stopped after $minutes min."
            drawFocusStats()
            drawDial()
        }
    }

    /** Stop without fuss: past halfway it still counts. */
    private fun stopBlock() = finishBlock(kept = DayPlanner.gentleStopCounts(block(), System.currentTimeMillis()))

    private fun takeBreak() {
        val block = block() ?: return
        val now = System.currentTimeMillis()
        val rest = block.breakMinutes * 60_000L
        FocusNudges.cancelAll(this)
        FocusNudges.schedule(this, "back", "Break's over", now + rest)
        save({ doc ->
            DayStore.recordFocus(doc, now, block.minutes)
            doc.remove("block")
            doc.put("break_until", now + rest).put("break_ms", rest)
        }) {
            focusStatus?.text = ""
            drawFocusStats()
            drawDial()
        }
    }

    private fun distracted() {
        save({ doc -> doc.optJSONObject("block")?.let { it.put("interruptions", it.optInt("interruptions") + 1) } }) {
            focusStatus?.text = "Noted. Back to it when you're ready."
        }
    }

    private fun saveTask(field: EditText) {
        val value = field.text.toString().trim().take(200)
        if (value != doc.optString("task")) save({ it.put("task", value) }) { drawDial() }
    }

    private fun addSteps(field: EditText) {
        val added = DayPlanner.chunkTask(field.text.toString())
        if (added.isEmpty()) return
        val known = steps().map { it.text.lowercase(Locale.UK) }.toSet()
        val next = steps() + added.filter { it.lowercase(Locale.UK) !in known }.map { DayPlanner.Step(it, false) }
        field.setText("")
        save({ it.put("steps", DayPlanner.stepsToJson(next)) }) { drawSteps(); drawDial() }
    }

    private fun drawSteps() {
        val list = stepList ?: return
        list.removeAllViews()
        val steps = steps()
        val current = steps.indexOfFirst { !it.done }
        steps.forEachIndexed { index, step ->
            list.addView(CheckBox(this).apply {
                text = step.text
                isChecked = step.done
                textSize = 15f
                setTextColor(if (step.done) Color.rgb(120, 120, 128) else Color.rgb(236, 236, 240))
                buttonTintList = android.content.res.ColorStateList.valueOf(Color.rgb(211, 47, 47))
                paintFlags = if (step.done) paintFlags or Paint.STRIKE_THRU_TEXT_FLAG else paintFlags and Paint.STRIKE_THRU_TEXT_FLAG.inv()
                typeface = Typeface.create("sans-serif", if (index == current) Typeface.BOLD else Typeface.NORMAL)
                setOnCheckedChangeListener { _, on ->
                    val next = steps().toMutableList()
                    if (index < next.size) next[index] = next[index].copy(done = on)
                    save({ it.put("steps", DayPlanner.stepsToJson(next)) }) { list.post { drawSteps(); drawDial() } }
                }
            }, LinearLayout.LayoutParams(-1, dp(44)))
        }
        if (steps.any { it.done }) list.addView(text("Clear done", 13f).apply {
            gravity = Gravity.END
            setTextColor(Color.rgb(236, 104, 115))
            isClickable = true
            isFocusable = true
            setPadding(dp(6), dp(8), dp(6), dp(4))
            setOnClickListener {
                save({ it.put("steps", DayPlanner.stepsToJson(steps().filter { s -> !s.done })) }) { drawSteps() }
            }
        }, LinearLayout.LayoutParams(-1, -2))
    }

    private fun drawFocusStats() {
        if (focusStats.size < 3) return
        val day = DayPlanner.focusDay(DayStore.focusHistory(doc), ZonedDateTime.now())
        focusStats[0].text = "${day.minutes}m"
        focusStats[1].text = day.blocks.toString()
        focusStats[2].text = day.streak.toString()
    }

    private fun askForNotifications() {
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED) notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }

    // ── fitness ────────────────────────────────────────────────────────────

    private fun buildFitness(body: LinearLayout) {
        val now = ZonedDateTime.now()
        val workouts = DayStore.workouts(doc)
        val week = DayPlanner.week(workouts, now, DayStore.target(doc))

        val tiles = GridLayout(this).apply { columnCount = 2 }
        listOf(DayPlanner.clock(week.minutes.toDouble()) to "This week",
            "${week.workouts}/${week.target}" to "Workouts",
            "%.1f".format(Locale.UK, week.distanceKm).removeSuffix(".0") to "Kilometres",
            week.streak.toString() to "Day streak").forEachIndexed { i, (value, name) ->
            val tile = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(16), dp(12), dp(16), dp(12))
                background = rounded(Color.rgb(20, 20, 23), dp(16), Color.rgb(44, 44, 50))
            }
            tile.addView(text(value, 26f).apply {
                setTextColor(Color.WHITE)
                typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
            })
            tile.addView(text(name, 11f).apply { setTextColor(Color.rgb(140, 140, 148)) })
            tiles.addView(tile, GridLayout.LayoutParams(GridLayout.spec(i / 2), GridLayout.spec(i % 2, 1f)).apply {
                width = 0
                setMargins(if (i % 2 == 0) 0 else dp(5), 0, if (i % 2 == 0) dp(5) else 0, dp(10))
            })
        }
        body.addView(tiles, LinearLayout.LayoutParams(-1, -2))

        val chart = card(body)
        chart.addView(caption("This week"))
        chart.addView(WeekBarsView(this).apply { show(week.days, week.today) }, LinearLayout.LayoutParams(-1, dp(150)))

        val log = card(body)
        val kinds = LinearLayout(this)
        val distanceRow = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL }
        DayPlanner.KINDS.take(6).forEach { k ->
            kinds.addView(chip(DayPlanner.kindName(k), k == kind) {
                kind = k
                if (k !in DayPlanner.DISTANCE_KINDS) distance = 0.0
                show(FITNESS)
            }, LinearLayout.LayoutParams(-2, dp(36)).apply { rightMargin = dp(6) })
        }
        log.addView(HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false; addView(kinds) })
        log.addView(stepper("Time", { "$minutes min" }, { minutes = (minutes - 5).coerceAtLeast(5) },
            { minutes = (minutes + 5).coerceAtMost(600) }))
        if (kind in DayPlanner.DISTANCE_KINDS) log.addView(stepper("Distance",
            { if (distance <= 0) "None" else "%.1f km".format(Locale.UK, distance) },
            { distance = (distance - 0.5).coerceAtLeast(0.0) }, { distance = (distance + 0.5).coerceAtMost(200.0) }))
        val note = input("Note")
        log.addView(note, LinearLayout.LayoutParams(-1, dp(48)).apply { topMargin = dp(6) })
        log.addView(button("Log ${DayPlanner.kindName(kind).lowercase(Locale.UK)}", accent = true) {
            logWorkout(note.text.toString())
        }, LinearLayout.LayoutParams(-1, dp(50)).apply { topMargin = dp(8) })

        val goal = card(body)
        goal.addView(stepper("Workouts a week", { DayStore.target(doc).toString() },
            { saveTarget(DayStore.target(doc) - 1) }, { saveTarget(DayStore.target(doc) + 1) }))

        val recent = card(body)
        recent.addView(caption("Recent"))
        if (workouts.isEmpty()) recent.addView(text("Nothing logged yet.", 14f).apply { setTextColor(Color.rgb(130, 130, 138)) })
        workouts.take(6).forEach { workout ->
            val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; setPadding(0, dp(4), 0, dp(4)) }
            val words = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
            words.addView(text(DayPlanner.kindName(workout.kind), 15f).apply { setTextColor(Color.WHITE) })
            words.addView(text(DayPlanner.whenText(workout.at, now), 11f).apply { setTextColor(Color.rgb(130, 130, 138)) })
            row.addView(words, LinearLayout.LayoutParams(0, -2, 1f))
            row.addView(text(DayPlanner.amountText(workout), 13f).apply {
                typeface = Typeface.MONOSPACE
                setTextColor(Color.rgb(190, 190, 198))
            })
            row.addView(text("×", 22f).apply {
                gravity = Gravity.CENTER
                contentDescription = "Remove this ${DayPlanner.kindName(workout.kind)}"
                setTextColor(Color.rgb(150, 150, 158))
                isClickable = true
                isFocusable = true
                setOnClickListener { removeWorkout(workout) }
            }, LinearLayout.LayoutParams(dp(44), dp(44)))
            recent.addView(row)
        }
    }

    private fun stepper(name: String, value: () -> String, less: () -> Unit, more: () -> Unit): View {
        val row = LinearLayout(this).apply { gravity = Gravity.CENTER_VERTICAL; setPadding(0, dp(6), 0, 0) }
        row.addView(text(name, 15f).apply { setTextColor(Color.rgb(223, 226, 235)) }, LinearLayout.LayoutParams(0, -2, 1f))
        val shown = text(value(), 15f).apply { gravity = Gravity.CENTER; setTextColor(Color.WHITE) }
        row.addView(button("−") { less(); shown.text = value() }.apply { contentDescription = "Less $name" },
            LinearLayout.LayoutParams(dp(48), dp(44)))
        row.addView(shown, LinearLayout.LayoutParams(dp(84), -2))
        row.addView(button("+") { more(); shown.text = value() }.apply { contentDescription = "More $name" },
            LinearLayout.LayoutParams(dp(48), dp(44)))
        return row
    }

    private fun saveTarget(value: Int) {
        val next = value.coerceIn(1, 14)
        save({ it.put("weekly_target", next) }) { show(FITNESS) }
    }

    private fun logWorkout(note: String) {
        if (!loaded) return
        val k = kind
        val m = minutes.toDouble()
        val km = distance.takeIf { it > 0 && k in DayPlanner.DISTANCE_KINDS }
        io.execute {
            val result = runCatching { DayStore.addWorkout(this, k, m, km, note) to DayStore.read(this) }
            main.post {
                if (isDestroyed) return@post
                result.onSuccess { (workout, fresh) ->
                    doc = fresh
                    confirmTap(content)
                    if (tab == FITNESS) show(FITNESS)
                    say("${DayPlanner.kindName(workout.kind)} logged, ${DayPlanner.amountText(workout)}.")
                }.onFailure { say(it.message ?: "That workout was not saved.") }
            }
        }
    }

    private fun removeWorkout(workout: DayPlanner.Workout) {
        save({ doc -> doc.put("workouts", org.json.JSONArray().apply {
            DayStore.workouts(doc).filter { it.id != workout.id }.forEach { put(it.toJson()) } }) }) {
            show(FITNESS)
            Snackbar.make(content, "${DayPlanner.kindName(workout.kind)} removed.", Snackbar.LENGTH_LONG)
                .setAction("Undo") {
                    save({ doc -> doc.put("workouts", org.json.JSONArray().apply {
                        (DayStore.workouts(doc) + workout).sortedByDescending { it.at }.forEach { put(it.toJson()) } }) }) {
                        show(FITNESS)
                    }
                }.show()
        }
    }

    // ── shared ─────────────────────────────────────────────────────────────

    /** Change the stored day off the main thread (Keystore), then redraw with what was saved. */
    private fun save(change: (JSONObject) -> Unit, then: () -> Unit = {}) {
        io.execute {
            val result = runCatching { DayStore.update(this, change) }
            main.post {
                if (isDestroyed) return@post
                result.onSuccess { doc = it; then() }.onFailure { say(it.message ?: "That was not saved.") }
            }
        }
    }

    private fun say(message: String) = Snackbar.make(content, message, Snackbar.LENGTH_SHORT).show()

    /** A small tick under the thumb when a block starts or a workout is kept. */
    private fun confirmTap(view: View?) {
        view?.performHapticFeedback(if (Build.VERSION.SDK_INT >= 30) HapticFeedbackConstants.CONFIRM
            else HapticFeedbackConstants.VIRTUAL_KEY)
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
    private fun rounded(colour: Int, radius: Int, border: Int? = null) = GradientDrawable().apply {
        setColor(colour)
        cornerRadius = radius.toFloat()
        if (border != null) setStroke(dp(1), border)
    }
    private fun text(value: String, size: Float) = TextView(this).apply {
        text = value
        textSize = size
        setTextColor(Color.rgb(223, 226, 235))
        setPadding(0, dp(2), 0, dp(2))
    }
    private fun caption(value: String) = text(value, 12f).apply {
        setTextColor(Color.rgb(140, 140, 148))
        setPadding(0, 0, 0, dp(6))
    }
    private fun card(parent: LinearLayout, tight: Boolean = false) = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(16), dp(if (tight) 10 else 14), dp(16), dp(if (tight) 10 else 14))
        background = rounded(Color.rgb(20, 20, 23), dp(18), Color.rgb(44, 44, 50))
        parent.addView(this, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(if (tight) 8 else 12) })
    }
    private fun input(hint: String) = EditText(this).apply {
        this.hint = hint
        setTextColor(Color.WHITE)
        setHintTextColor(Color.rgb(130, 130, 138))
        textSize = 15f
        isSingleLine = true
        inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        background = rounded(Color.rgb(26, 26, 30), dp(14), Color.rgb(55, 55, 62))
        setPadding(dp(14), dp(8), dp(14), dp(8))
    }
    private fun button(label: String, accent: Boolean = false, action: () -> Unit) = Button(this).apply {
        text = label
        isAllCaps = false
        minHeight = dp(44)
        setTextColor(Color.WHITE)
        backgroundTintList = android.content.res.ColorStateList.valueOf(
            if (accent) Color.rgb(164, 32, 42) else Color.rgb(42, 42, 48))
        setOnClickListener { action() }
    }
    private fun chip(label: String, on: Boolean, action: () -> Unit) = text(label, 13f).apply {
        gravity = Gravity.CENTER
        setPadding(dp(14), 0, dp(14), 0)
        setTextColor(if (on) Color.WHITE else Color.rgb(176, 176, 184))
        background = rounded(if (on) Color.rgb(64, 28, 35) else Color.rgb(27, 27, 31), dp(18),
            if (on) Color.rgb(150, 44, 54) else Color.rgb(52, 52, 59))
        isClickable = true
        isFocusable = true
        contentDescription = if (on) "$label, selected" else label
        setOnClickListener { action() }
    }
}
