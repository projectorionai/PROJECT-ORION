package com.orion.companion

/** Shared request/output revision: a delayed stop belongs to one foreground/background reply. */
internal object SpeechPlaybackEpoch {
    private var revision = 0L
    @Synchronized fun begin(): Long = ++revision
    @Synchronized fun current(): Long = revision
    @Synchronized fun isCurrent(expected: Long): Boolean = expected == revision
}
