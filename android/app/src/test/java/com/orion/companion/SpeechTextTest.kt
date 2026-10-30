package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class SpeechTextTest {
    @Test fun keepsTheFinalSentenceOfAComprehensiveReply() {
        val reply = (1..250).joinToString(" ") { "This is section $it of the full answer." } + " FINAL WORDS."
        val parts = SpeechText.parts(reply)
        assertTrue(parts.size > 5)
        assertTrue(parts.all { it.length <= 1200 })
        assertEquals(reply, parts.joinToString(" "))
        assertTrue(parts.last().endsWith("FINAL WORDS."))
    }

    @Test fun currentSpeechModelsPrecedeTheExhaustedPreview() {
        val models = SpeechText.voiceModels("gemini-3.1-flash-tts-preview")
        assertEquals("gemini-3.8-flash-tts", models.first())
        assertEquals("gemini-3.1-flash-tts-preview", models.last())
        assertEquals("gemini-3.8-flash-lite-tts", SpeechText.voiceModels("gemini-3.8-flash-lite-tts").first())
        assertEquals(models, SpeechText.voiceModels("unexpected-model"))
    }

    @Test fun neverSplitsSurrogatePairsOrDropsLongTokens() {
        val reply = "a".repeat(899) + "😀" + "b".repeat(1900)
        val parts = SpeechText.parts(reply)
        assertEquals(reply, parts.joinToString(""))
        assertTrue(parts.none { it.last().isHighSurrogate() || it.first().isLowSurrogate() })
    }

    @Test fun favoursCompleteSentencesAndIgnoresBlankSpeech() {
        val reply = "First complete sentence. " + "word ".repeat(200) + "The end."
        val parts = SpeechText.parts(reply, 50)
        assertEquals("First complete sentence.", parts.first())
        assertEquals(reply.trim().replace(Regex("\\s+"), " "), parts.joinToString(" "))
        assertTrue(SpeechText.parts(" \n ").isEmpty())
    }

    @Test fun streamedPhrasesStartAtCompletedSentencesAndKeepTheTail() {
        val stream = SpeechText.Stream()
        assertTrue(stream.add("This is a complete opening sentence").isEmpty())
        assertEquals(listOf("This is a complete opening sentence."),
            stream.add(". The next sentence arrives in two"))
        assertTrue(stream.add(" chunks").isEmpty())
        assertEquals("The next sentence arrives in two chunks", stream.finish())
        assertEquals("", stream.finish())
    }

    @Test fun streamingSpeechWaitsForRequestsThatCouldRunActions() {
        assertTrue(SpeechText.mayStreamAnswer("Why does the moon change shape?"))
        assertFalse(SpeechText.mayStreamAnswer("Can you open Calculator?"))
        assertFalse(SpeechText.mayStreamAnswer("Please e-mail Alex about tomorrow."))
    }

    @Test fun aLongUnpunctuatedAnswerContinuesWithoutLosingWords() {
        val reply = (1..150).joinToString(" ") { "word$it" }
        val stream = SpeechText.Stream()
        val spoken = mutableListOf<String>()
        reply.chunked(17).forEach { spoken += stream.add(it) }
        spoken += stream.finish()
        assertTrue(spoken.size > 1)
        assertEquals(reply, spoken.joinToString(" "))
    }

    @Test fun paragraphChangesDoNotCreateSeparateTinySpeechRequests() {
        val reply = (1..12).joinToString("\n\n") { "This is paragraph $it, with a short complete sentence." }
        val stream = SpeechText.Stream()
        val spoken = mutableListOf<String>()
        reply.chunked(9).forEach { spoken += stream.add(it) }
        spoken += stream.finish()
        assertTrue(spoken.size < 6)
        assertTrue(spoken.none { it.contains('\n') })
        assertEquals(reply.replace(Regex("\\s+"), " "), spoken.joinToString(" "))
        assertEquals(reply.replace(Regex("\\s+"), " "), SpeechText.parts(reply).joinToString(" "))
    }
}
