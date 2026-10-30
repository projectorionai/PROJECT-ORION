package com.orion.companion

import java.io.Closeable
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/** One ordered PCM output for an entire reply, independent of synthesis requests. */
internal class PcmReplyPlayback(
    private val cancelled: () -> Boolean,
    private val write: (ByteArray) -> Unit,
    private val drain: () -> Unit,
    private val release: () -> Unit,
) : Closeable {
    // Up to roughly 175 seconds of mono PCM16. Back-pressure preserves every
    // sample whilst preventing a fast provider from growing memory indefinitely.
    private val pending = ArrayBlockingQueue<ByteArray>(256)
    private val stopped = AtomicBoolean()
    private val finished = CountDownLatch(1)
    private val queuedBytes = AtomicLong()
    private val acceptedBytes = AtomicLong()
    @Volatile private var failure: Throwable? = null
    @Volatile private var ended = false
    val hasAudio: Boolean get() = acceptedBytes.get() > 0L

    private val worker = Thread({
        try {
            while (!stopped.get() && !cancelled()) {
                val pcm = pending.poll(50, TimeUnit.MILLISECONDS) ?: continue
                if (pcm.isEmpty()) break
                if (stopped.get() || cancelled()) break
                write(pcm)
                queuedBytes.addAndGet(-pcm.size.toLong())
            }
            if (!stopped.get() && !cancelled()) drain()
        } catch (problem: Throwable) {
            failure = problem
        } finally {
            stopped.set(true)
            pending.clear()
            runCatching(release)
            finished.countDown()
        }
    }, "orion-phone-pcm").apply { isDaemon = true; start() }

    fun append(pcm: ByteArray) {
        require(pcm.isNotEmpty() && pcm.size % 2 == 0)
        check(!ended) { "Speech output has already finished" }
        for (offset in pcm.indices step 32_768) {
            val part = pcm.copyOfRange(offset, minOf(offset + 32_768, pcm.size))
            queuedBytes.addAndGet(part.size.toLong())
            var submitted = false
            try {
                while (!stopped.get() && !cancelled()) {
                    check(failure == null) { "Speech output failed" }
                    if (pending.offer(part, 50, TimeUnit.MILLISECONDS)) {
                        acceptedBytes.addAndGet(part.size.toLong())
                        submitted = true
                        break
                    }
                }
                check(submitted) { "Speech output stopped" }
            } finally {
                if (!submitted) queuedBytes.addAndGet(-part.size.toLong())
            }
        }
    }

    /** Finish once after the final section, allowing all queued samples to play. */
    fun finish() {
        if (!ended) {
            ended = true
            while (!stopped.get() && !cancelled() &&
                !pending.offer(ByteArray(0), 50, TimeUnit.MILLISECONDS)) Unit
        }
        // Include the output device's remaining buffer and a scheduling margin.
        // Already played sections never inflate the wait for a long answer.
        val waitMillis = queuedBytes.get().coerceAtLeast(0L) * 1000L / 48_000L + 10_000L
        val deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(waitMillis)
        while (!cancelled() && System.nanoTime() < deadline) {
            if (finished.await(50, TimeUnit.MILLISECONDS)) {
                check(failure == null) { "Speech output failed" }
                return
            }
        }
        check(cancelled()) { "Speech output did not finish" }
    }

    override fun close() {
        stopped.set(true)
        pending.clear()
        pending.offer(ByteArray(0))
        worker.interrupt()
        if (Thread.currentThread() !== worker) finished.await(500, TimeUnit.MILLISECONDS)
    }
}
