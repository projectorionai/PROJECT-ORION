package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhonePromptWindowTest {
    @Test fun keepsCurrentQuestionAndTailOfLongPreviousReply() {
        val history = listOf(
            PhonePromptLine("user", "Explain the whole topic"),
            PhonePromptLine("model", "A".repeat(70_000) + "ENDING"),
            PhonePromptLine("user", "Continue from where you stopped"),
        )
        val selected = PhonePromptWindow.select(history)
        assertEquals("Continue from where you stopped", selected.last().text)
        assertTrue(selected[1].text.startsWith("[Earlier reply omitted]"))
        assertTrue(selected[1].text.endsWith("ENDING"))
        assertTrue(selected.sumOf { it.text.length } <= PhonePromptWindow.MAX_CHARS)
    }

    @Test fun capsManyOldMessagesWithoutTrimmingTheCurrentRequest() {
        val history = (0..50).map { PhonePromptLine(if (it % 2 == 0) "user" else "model", "x".repeat(4_000)) } +
            PhonePromptLine("user", "latest request")
        val selected = PhonePromptWindow.select(history)
        assertEquals("latest request", selected.last().text)
        assertTrue(selected.size <= 16)
        assertTrue(selected.sumOf { it.text.length } <= PhonePromptWindow.MAX_CHARS)
    }
}
