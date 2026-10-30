package com.orion.companion

import android.animation.ValueAnimator
import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.MotionEvent
import android.view.View
import android.view.animation.OvershootInterpolator
import kotlin.math.max
import kotlin.math.min

/** A drag-only approval control; a tap cannot execute an agent action. */
class SlideToConfirmView(context: Context) : View(context) {
    var onConfirmed: (() -> Unit)? = null
    var label = "SLIDE TO AUTHORISE"
        set(value) { field = value; invalidate() }

    private val density = resources.displayMetrics.density
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF301216.toInt() }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFF801818.toInt() }
    private val thumb = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xFFD32F2F.toInt() }
    private val copy = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFF2E6E8.toInt()
        textSize = 11f * resources.displayMetrics.scaledDensity
        typeface = android.graphics.Typeface.MONOSPACE
        textAlign = Paint.Align.CENTER
    }
    private val bounds = RectF()
    private var progress = 0f
    private var dragging = false
    private var completed = false
    private var downX = 0f
    private var resetAnimator: ValueAnimator? = null

    init {
        isClickable = true
        contentDescription = "Drag the crimson slider all the way right to authorize"
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        setMeasuredDimension(MeasureSpec.getSize(widthMeasureSpec), (56f * density).toInt())
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val inset = 2f * density
        bounds.set(inset, inset, width - inset, height - inset)
        val radius = bounds.height() / 2f
        canvas.drawRoundRect(bounds, radius, radius, track)
        val travel = max(0f, bounds.width() - 2f * radius)
        canvas.drawRoundRect(bounds.left, bounds.top, bounds.left + 2f * radius + travel * progress,
            bounds.bottom, radius, radius, fill)
        val cx = bounds.left + radius + travel * progress
        canvas.drawCircle(cx, bounds.centerY(), radius - 3f * density, thumb)
        copy.textAlign = Paint.Align.CENTER
        canvas.drawText("››", cx, bounds.centerY() - (copy.ascent() + copy.descent()) / 2f, copy)
        copy.textAlign = Paint.Align.LEFT
        val labelX = bounds.left + 2f * radius + 9f * density
        if (labelX + copy.measureText(label) < bounds.right - 9f * density) {
            canvas.drawText(label, labelX, bounds.centerY() - (copy.ascent() + copy.descent()) / 2f, copy)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (completed) return true
        val radius = height / 2f
        val travel = max(1f, width - 2f * radius)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                if (event.x > radius * 2f || event.y < 0f || event.y > height) return false
                resetAnimator?.cancel()
                downX = event.x
                dragging = true
                parent?.requestDisallowInterceptTouchEvent(true)
                return true
            }
            MotionEvent.ACTION_POINTER_DOWN, MotionEvent.ACTION_POINTER_UP -> {
                dragging = false
                springBack()
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (!dragging) return false
                progress = min(1f, max(0f, (event.x - radius) / travel))
                invalidate()
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (!dragging) return false
                dragging = false
                progress = min(1f, max(0f, (event.x - radius) / travel))
                if (progress >= 0.92f && event.x - downX >= 96f * density) {
                    progress = 1f
                    completed = true
                    invalidate()
                    hapticClick()
                    onConfirmed?.invoke()
                } else springBack()
                return true
            }
            MotionEvent.ACTION_CANCEL -> {
                dragging = false
                springBack()
                return true
            }
        }
        return false
    }

    private fun springBack() {
        resetAnimator?.cancel()
        resetAnimator = ValueAnimator.ofFloat(progress, 0f).apply {
            duration = 320
            interpolator = OvershootInterpolator(0.8f)
            addUpdateListener { progress = (it.animatedValue as Float).coerceIn(0f, 1f); invalidate() }
            start()
        }
    }

    private fun hapticClick() {
        val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            vibrator.vibrate(VibrationEffect.createPredefined(VibrationEffect.EFFECT_CLICK))
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(25)
        }
    }

    override fun onDetachedFromWindow() {
        resetAnimator?.cancel()
        super.onDetachedFromWindow()
    }
}
