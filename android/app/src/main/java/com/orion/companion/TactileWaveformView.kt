package com.orion.companion

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.view.View
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

/** Live audio waveform for listening and speaking, with a quiet idle line. */
class TactileWaveformView(context: Context) : View(context) {
    var amplitude = 0f
        set(value) { field = value.coerceIn(0f, 1f); invalidate() }
    var watching = false
        set(value) { field = value; invalidate() }
    var speaking = false
        set(value) { field = value; invalidate() }

    private val density = resources.displayMetrics.density
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        strokeWidth = 1.6f * density
    }
    private val glow = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        strokeWidth = 5f * density
    }
    private val baseline = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0x44525258
        strokeWidth = density
    }
    private val path = Path()
    private val started = android.os.SystemClock.uptimeMillis()
    private var displayed = 0f

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        setMeasuredDimension(resolveSize((240f * density).toInt(), widthMeasureSpec),
            resolveSize((76f * density).toInt(), heightMeasureSpec))
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (width <= 0 || height <= 0) return
        val seconds = (android.os.SystemClock.uptimeMillis() - started) / 1000f
        val target = when {
            speaking -> 0.68f
            watching -> max(0.07f, sqrt(amplitude * 8f).coerceAtMost(1f))
            else -> 0.02f
        }
        displayed += (target - displayed) * 0.22f
        val centre = height / 2f
        val edge = 8f * density
        val widthAvailable = (width - edge * 2).coerceAtLeast(1f)
        canvas.drawLine(edge, centre, width - edge, centre, baseline)

        val colour = if (speaking) 0xFFD32F2F.toInt() else 0xFFA4202A.toInt()
        for (layer in 0..2) {
            path.reset()
            val shift = layer * 0.78f
            val heightScale = (height * 0.39f) * displayed * (1f - layer * 0.19f)
            for (point in 0..95) {
                val t = point / 95f
                val envelope = sin(PI.toFloat() * t).let { it * it }
                val oscillation = sin(t * (26f + layer * 3f) - seconds *
                    (if (speaking) 9f else 4.5f) + shift)
                val accent = sin(t * 57f + seconds * 3.2f - shift) * 0.19f
                val x = edge + t * widthAvailable
                val y = centre + (oscillation + accent) * heightScale * envelope
                if (point == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            glow.color = colour
            glow.alpha = if (layer == 0) 50 else 25
            canvas.drawPath(path, glow)
            line.color = colour
            line.alpha = 235 - layer * 64
            canvas.drawPath(path, line)
        }
        // Narrow pulses around the centre make quiet speech visible immediately.
        if (watching && amplitude > 0.004f && !speaking) {
            val pulse = sqrt(amplitude * 8f).coerceAtMost(1f) * height * 0.2f
            line.alpha = 175
            for (index in -5..5) {
                val x = width / 2f + index * 5f * density
                val taper = 1f - abs(index) / 6f
                canvas.drawLine(x, centre - pulse * taper, x, centre + pulse * taper, line)
            }
        }
        if ((watching || speaking || displayed > 0.025f) && isAttachedToWindow)
            postInvalidateDelayed(33)
    }
}
