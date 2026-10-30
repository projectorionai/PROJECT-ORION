package com.orion.companion

import java.util.concurrent.CountDownLatch

/** An utterance stays active until the engine completes it or its owner cancels. */
internal class DeviceSpeechTurn(val token: String, val firstSound: () -> Unit = {}) {
    val done = CountDownLatch(1)
    @Volatile var success = false
}

/** Late Android speech callbacks cannot complete or release a newer utterance. */
internal class DeviceSpeechTurns {
    @Volatile var current: DeviceSpeechTurn? = null
        private set
    @Synchronized fun begin(turn: DeviceSpeechTurn) {
        current?.done?.countDown()
        current = turn
    }
    @Synchronized fun complete(token: String, success: Boolean) {
        current?.takeIf { it.token == token }?.let {
            it.success = success
            it.done.countDown()
        }
    }
    @Synchronized fun clear(turn: DeviceSpeechTurn) {
        if (current === turn) current = null
    }
    @Synchronized fun cancel() {
        current?.done?.countDown()
        current = null
    }
}
