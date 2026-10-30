package com.orion.companion

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import btools.routingapp.IBRouterService
import java.io.IOException
import java.util.concurrent.Executors

/** Calls BRouter's local, public Android service. Routing is never performed on the UI thread. */
internal class OfflineCarRouter(context: Context) {
    private val app = context.applicationContext
    private val main = Handler(Looper.getMainLooper())
    private val worker = Executors.newSingleThreadExecutor { task ->
        Thread(task, "orion-offline-car-router").apply { isDaemon = true }
    }
    private var current: Job? = null
    private var ipcInFlight = false
    private var closed = false

    private class Job(val callback: (Result<String>) -> Unit) {
        var completed = false
        var bound = false
        lateinit var connection: ServiceConnection
        lateinit var timeout: Runnable
    }

    fun isInstalled(): Boolean = runCatching {
        app.packageManager.getServiceInfo(COMPONENT, 0).enabled
    }.getOrDefault(false)

    fun calculate(start: OfflineCoordinate, destination: OfflineCoordinate,
        callback: (Result<String>) -> Unit) {
        check(Looper.myLooper() == Looper.getMainLooper())
        if (closed) { callback(Result.failure(IOException("Route screen is closed"))); return }
        if (current != null || ipcInFlight) {
            callback(Result.failure(IOException("A route calculation is still running"))); return
        }
        if (!isInstalled()) { callback(Result.failure(IOException("Install BRouter and download the journey's routing regions first"))); return }
        if (OfflineRoute.distance(start, destination) < 10) {
            callback(Result.failure(IOException("The destination is already within 10 metres"))); return
        }
        val job = Job(callback)
        current = job
        job.timeout = Runnable { finish(job, Result.failure(IOException("Route calculation timed out. Shorten the journey or check BRouter's downloaded regions."))) }
        job.connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, binder: IBinder) {
                if (current !== job || job.completed || closed) return
                val service = IBRouterService.Stub.asInterface(binder)
                ipcInFlight = true
                worker.execute {
                    val result = runCatching {
                        val parameters = Bundle().apply {
                            putDoubleArray("lats", doubleArrayOf(start.latitude, destination.latitude))
                            putDoubleArray("lons", doubleArrayOf(start.longitude, destination.longitude))
                            putString("profile", "car-fast")
                            putString("v", "motorcar")
                            putString("fast", "1")
                            putString("format", "gpx")
                            putString("trackFormat", "gpx")
                            putString("turnInstructionFormat", "osmand")
                            putString("timode", "3")
                            putString("maxRunningTime", "30")
                            putString("acceptCompressedResult", "true")
                            // No pathToFileResult: Android scoped storage returns the route via Binder.
                        }
                        OfflineRouteGpx.serviceResult(service.getTrackFromParams(parameters)).also {
                            OfflineRouteGpx.parse(it) // A textual BRouter error never becomes a route.
                        }
                    }
                    main.post { ipcInFlight = false; finish(job, result) }
                }
            }
            override fun onServiceDisconnected(name: ComponentName) {
                finish(job, Result.failure(IOException("BRouter disconnected during calculation")))
            }
            override fun onNullBinding(name: ComponentName) {
                finish(job, Result.failure(IOException("BRouter could not open its route service")))
            }
            override fun onBindingDied(name: ComponentName) {
                finish(job, Result.failure(IOException("BRouter's route service stopped")))
            }
        }
        try {
            job.bound = app.bindService(Intent().setComponent(COMPONENT), job.connection, Context.BIND_AUTO_CREATE)
            if (!job.bound) finish(job, Result.failure(IOException("BRouter's route service is unavailable")))
            else main.postDelayed(job.timeout, 45_000)
        } catch (error: Exception) { finish(job, Result.failure(IOException("Could not connect to BRouter", error))) }
    }

    /** A timed-out remote Binder call cannot be cancelled; its eventual reply is discarded. */
    fun cancel() {
        current?.let { finish(it, Result.failure(IOException("Route calculation cancelled"))) }
    }

    fun close() { closed = true; cancel(); worker.shutdownNow() }

    private fun finish(job: Job, result: Result<String>) {
        if (job.completed) return
        job.completed = true
        main.removeCallbacks(job.timeout)
        if (job.bound) { runCatching { app.unbindService(job.connection) }; job.bound = false }
        if (current === job) current = null
        if (!closed) job.callback(result)
    }

    companion object {
        const val PACKAGE = "btools.routingapp"
        private val COMPONENT = ComponentName(PACKAGE, "btools.routingapp.BRouterService")
    }
}
