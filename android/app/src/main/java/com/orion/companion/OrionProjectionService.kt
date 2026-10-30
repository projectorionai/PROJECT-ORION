package com.orion.companion

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Point
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.SystemClock
import android.view.Display
import androidx.core.content.ContextCompat
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Captures the screen only after an Activity has obtained a fresh MediaProjection
 * consent result. The result Intent is kept in memory for this service start only.
 */
class OrionProjectionService : Service() {

    private lateinit var captureThread: HandlerThread
    private lateinit var captureHandler: Handler
    private var projection: MediaProjection? = null
    private var projectionCallback: MediaProjection.Callback? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var lastFrameAtMs = 0L
    @Volatile private var stopping = false

    override fun onCreate() {
        super.onCreate()
        captureThread = HandlerThread("orion-screen-capture").also { it.start() }
        captureHandler = Handler(captureThread.looper)
        createNotificationChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopping = true
            isCapturing = false
            phoneFrames.clear()
            stopSelf()
            return START_NOT_STICKY
        }
        if (intent?.action != ACTION_START) {
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
        val resultData = intent.projectionData()
        if (resultCode != Activity.RESULT_OK || resultData == null) {
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        stopping = false
        phoneFrames.clear()

        try {
            // Android 14 requires the foreground service to be running before
            // getMediaProjection() uses the one-time consent result.
            val notification = captureNotification()
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION,
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (_: RuntimeException) {
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        captureHandler.post {
            if (stopping) return@post
            releaseCapture()
            if (stopping) return@post
            try {
                beginCapture(resultCode, resultData)
                if (stopping) releaseCapture()
            } catch (_: RuntimeException) {
                releaseCapture()
                stopSelfResult(startId)
            }
        }
        return START_NOT_STICKY
    }

    private fun beginCapture(resultCode: Int, resultData: Intent) {
        val manager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val newProjection = manager.getMediaProjection(resultCode, resultData)
            ?: throw IllegalStateException("MediaProjection consent was not granted")
        projection = newProjection
        val callback = object : MediaProjection.Callback() {
            override fun onStop() {
                if (projection !== newProjection) return
                releaseCapture()
                stopSelf()
            }

            override fun onCapturedContentResize(width: Int, height: Int) {
                if (projection === newProjection) resizeCapture(width, height)
            }
        }
        projectionCallback = callback
        newProjection.registerCallback(callback, captureHandler)

        val screen = (getSystemService(DISPLAY_SERVICE) as DisplayManager)
            .getDisplay(Display.DEFAULT_DISPLAY)
            ?: throw IllegalStateException("No screen display is available")
        val realSize = Point()
        @Suppress("DEPRECATION")
        screen.getRealSize(realSize)
        val captureSize = scaledSize(realSize.x, realSize.y)
        lastFrameAtMs = 0L
        val reader = newReader(captureSize.x, captureSize.y)
        imageReader = reader
        virtualDisplay = newProjection.createVirtualDisplay(
            "ORION Ocular Uplink",
            captureSize.x,
            captureSize.y,
            resources.displayMetrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            captureHandler,
        )
        isCapturing = true
    }

    private fun resizeCapture(width: Int, height: Int) {
        if (width <= 0 || height <= 0) return
        val display = virtualDisplay ?: return
        val size = scaledSize(width, height)
        val oldReader = imageReader ?: return
        if (oldReader.width == size.x && oldReader.height == size.y) return
        var replacement: ImageReader? = null
        try {
            val nextReader = newReader(size.x, size.y)
            replacement = nextReader
            display.resize(size.x, size.y, resources.displayMetrics.densityDpi)
            display.setSurface(nextReader.surface)
            imageReader = nextReader
            oldReader.setOnImageAvailableListener(null, null)
            oldReader.close()
        } catch (_: RuntimeException) {
            runCatching { replacement?.close() }
            releaseCapture()
            stopSelf()
        }
    }

    private fun newReader(width: Int, height: Int): ImageReader =
        ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2).also { reader ->
            reader.setOnImageAvailableListener({ available ->
                val image = try {
                    available.acquireLatestImage()
                } catch (_: IllegalStateException) {
                    null
                } ?: return@setOnImageAvailableListener
                try {
                    val now = SystemClock.elapsedRealtime()
                    val phoneMode = PhoneAiProfile.isPhoneMode(this)
                    val desktopConnected = !phoneMode && OrionFrameTransport.canSendFrames()
                    if (now - lastFrameAtMs >= MIN_FRAME_INTERVAL_MS && isCapturing &&
                        (phoneMode || desktopConnected)) {
                        lastFrameAtMs = now
                        val jpeg = try {
                            encodeJpeg(image)
                        } catch (_: RuntimeException) {
                            null
                        }
                        jpeg?.takeIf { isCapturing && !stopping }?.let {
                            // Local evidence is read only for an explicit screen request.
                            // Its provider upload is separate from the capture permission.
                            if (phoneMode) phoneFrames.put(it, now)
                            if (desktopConnected) runCatching { OrionFrameTransport.sendFrame(it) }
                        }
                    }
                } finally {
                    image.close()
                }
            }, captureHandler)
        }

    private fun encodeJpeg(image: Image): ByteArray? {
        val plane = image.planes.firstOrNull() ?: return null
        if (plane.pixelStride != BYTES_PER_PIXEL) return null
        val width = image.width
        val height = image.height
        val packedRowBytes = width * BYTES_PER_PIXEL
        val buffer = plane.buffer.duplicate()
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            if (plane.rowStride == packedRowBytes) {
                buffer.position(0)
                bitmap.copyPixelsFromBuffer(buffer)
            } else {
                val packed = ByteArray(packedRowBytes * height)
                for (row in 0 until height) {
                    buffer.position(row * plane.rowStride)
                    buffer.get(packed, row * packedRowBytes, packedRowBytes)
                }
                bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(packed))
            }
            return ByteArrayOutputStream().use { output ->
                if (bitmap.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, output)) {
                    output.toByteArray()
                } else {
                    null
                }
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun releaseCapture() {
        isCapturing = false
        phoneFrames.clear()
        OrionFrameTransport.clearViewport()
        runCatching { imageReader?.setOnImageAvailableListener(null, null) }
        runCatching { virtualDisplay?.release() }
        runCatching { imageReader?.close() }
        val activeProjection = projection
        projectionCallback?.let { callback ->
            runCatching { activeProjection?.unregisterCallback(callback) }
        }
        runCatching { activeProjection?.stop() }
        virtualDisplay = null
        imageReader = null
        projection = null
        projectionCallback = null
    }

    override fun onDestroy() {
        stopping = true
        isCapturing = false
        phoneFrames.clear()
        captureHandler.post {
            releaseCapture()
            captureThread.quitSafely()
        }
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "ORION screen capture",
            NotificationManager.IMPORTANCE_LOW,
        )
        channel.description = "Shows when ORION can view your screen"
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun captureNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, if (PhoneAiProfile.isPhoneMode(this)) PhoneConsoleActivity::class.java
                else MainActivity::class.java).putExtra("desktop_mode", !PhoneAiProfile.isPhoneMode(this)),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            1,
            Intent(this, OrionProjectionService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        return Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_stat_orion)
            .setContentTitle("ORION is viewing your screen")
            .setContentText("Tap Stop to end screen sharing")
            .setContentIntent(openApp)
            .setColor(Color.rgb(211, 47, 47))
            .setOngoing(true)
            .setCategory(Notification.CATEGORY_SERVICE)
            .addAction(android.R.drawable.ic_media_pause, "Stop", stop)
            .build()
    }

    private fun scaledSize(width: Int, height: Int): Point {
        val scale = minOf(1f, MAX_CAPTURE_EDGE.toFloat() / max(width, height).coerceAtLeast(1))
        return Point(max(1, (width * scale).roundToInt()), max(1, (height * scale).roundToInt()))
    }

    @Suppress("DEPRECATION")
    private fun Intent.projectionData(): Intent? =
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getParcelableExtra(EXTRA_RESULT_DATA, Intent::class.java)
        } else {
            getParcelableExtra(EXTRA_RESULT_DATA)
        }

    companion object {
        private const val ACTION_START = "com.orion.companion.action.START_PROJECTION"
        private const val ACTION_STOP = "com.orion.companion.action.STOP_PROJECTION"
        private const val EXTRA_RESULT_CODE = "projection_result_code"
        private const val EXTRA_RESULT_DATA = "projection_result_data"
        private const val CHANNEL_ID = "orion_screen_capture"
        private const val NOTIFICATION_ID = 1101
        private const val MAX_CAPTURE_EDGE = 960
        private const val BYTES_PER_PIXEL = 4
        private const val JPEG_QUALITY = 65
        private const val MIN_FRAME_INTERVAL_MS = 500L
        private val phoneFrames = PhoneFrameCache()

        @Volatile
        var isCapturing: Boolean = false
            private set

        /** A defensive copy of the latest consented JPEG, never older than ten seconds. */
        fun latestFrame(): ByteArray? =
            if (isCapturing) phoneFrames.get(SystemClock.elapsedRealtime()) else null

        /** Call from a visible Activity after a fresh MediaProjection consent result. */
        fun start(context: Context, resultCode: Int, resultData: Intent) {
            if (resultCode != Activity.RESULT_OK) return
            val service = Intent(context, OrionProjectionService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_RESULT_CODE, resultCode)
                .putExtra(EXTRA_RESULT_DATA, resultData)
            ContextCompat.startForegroundService(context, service)
        }

        fun stop(context: Context) {
            phoneFrames.clear()
            context.stopService(Intent(context, OrionProjectionService::class.java))
        }
    }
}
