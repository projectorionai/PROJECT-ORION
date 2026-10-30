package com.orion.companion

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.text.TextPaint
import android.text.TextUtils
import android.view.View
import java.time.LocalDate
import java.time.format.TextStyle
import java.util.Locale

/** The focus countdown: one ring for what is left, the time, and the step being worked on. */
internal class FocusDialView(context: Context) : View(context) {
    var dial = DayPlanner.Dial("idle", 0, 0f, "25:00", "")
        set(value) {
            if (value == field) return
            field = value
            contentDescription = listOf(value.text, value.caption).filter { it.isNotBlank() }.joinToString(", ")
            invalidate()
        }
    private val density = resources.displayMetrics.density
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; color = Color.argb(24, 255, 255, 255)
    }
    private val arc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
    }
    private val time = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; typeface = Typeface.create("sans-serif-light", Typeface.NORMAL)
        fontFeatureSettings = "tnum"
    }
    private val caption = TextPaint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER; color = Color.rgb(160, 160, 168)
    }
    private val oval = RectF()

    init { contentDescription = "Focus timer" }

    override fun onDraw(canvas: Canvas) {
        val side = minOf(width, height) - 20 * density
        if (side < 60 * density) return
        val ring = maxOf(6 * density, side * 0.035f)
        val cx = width / 2f
        val cy = height / 2f
        oval.set(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
        track.strokeWidth = ring
        canvas.drawOval(oval, track)
        if (dial.mode != "idle") {
            arc.strokeWidth = ring
            arc.color = when (dial.mode) {
                "due" -> Color.rgb(211, 47, 47)
                "break" -> Color.rgb(120, 120, 128)
                else -> Color.WHITE
            }
            // What is left, shrinking clockwise from the top.
            val sweep = if (dial.mode == "due") 360f else 360f * (1f - dial.progress.coerceIn(0f, 1f))
            canvas.drawArc(oval, -90f, sweep, false, arc)
        }
        time.color = if (dial.mode == "idle") Color.rgb(200, 200, 206) else Color.WHITE
        time.textSize = side * 0.21f
        canvas.drawText(dial.text, cx, cy + time.textSize * 0.32f, time)
        if (dial.caption.isNotBlank()) {
            caption.textSize = maxOf(13 * density, side * 0.055f)
            val line = TextUtils.ellipsize(dial.caption, caption, side * 0.66f, TextUtils.TruncateAt.END)
            canvas.drawText(line, 0, line.length, cx, cy + side * 0.22f, caption)
        }
    }
}

/** This week, Monday to Sunday: minutes moved each day, today in crimson. */
internal class WeekBarsView(context: Context) : View(context) {
    private var days: List<Pair<LocalDate, Double>> = emptyList()
    private var today: LocalDate? = null
    private val density = resources.displayMetrics.density
    private val bar = Paint(Paint.ANTI_ALIAS_FLAG)
    private val label = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER }
    private val box = RectF()

    fun show(days: List<Pair<LocalDate, Double>>, today: LocalDate) {
        this.days = days
        this.today = today
        contentDescription = "This week: " + days.joinToString(", ") { (day, minutes) ->
            "${day.dayOfWeek.getDisplayName(TextStyle.SHORT, Locale.UK)} ${DayPlanner.clock(minutes)}"
        }
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (days.isEmpty()) return
        val top = 22 * density
        val bottom = height - 22 * density
        val slot = width / days.size.toFloat()
        val most = maxOf(30.0, days.maxOf { it.second })
        val radius = 6 * density
        label.textSize = 11 * density
        days.forEachIndexed { i, (day, minutes) ->
            val cx = slot * i + slot / 2
            val w = minOf(slot * 0.5f, 34 * density)
            val isToday = day == today
            if (minutes <= 0) {
                bar.color = Color.argb(40, 255, 255, 255)
                box.set(cx - w / 2, bottom - 2 * density, cx + w / 2, bottom)
                canvas.drawRoundRect(box, density, density, bar)
            } else {
                val h = maxOf(6 * density, ((bottom - top) * (minutes / most)).toFloat())
                bar.color = if (isToday) Color.rgb(211, 47, 47) else Color.rgb(64, 64, 70)
                box.set(cx - w / 2, bottom - h, cx + w / 2, bottom)
                canvas.drawRoundRect(box, radius, radius, bar)
                label.color = Color.rgb(176, 176, 184)
                canvas.drawText(DayPlanner.clock(minutes), cx, bottom - h - 6 * density, label)
            }
            label.color = if (isToday) Color.WHITE else Color.rgb(130, 130, 138)
            canvas.drawText(day.dayOfWeek.getDisplayName(TextStyle.SHORT, Locale.UK), cx,
                height - 5 * density, label)
        }
    }
}
