package com.orion.companion

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.ZoneId
import java.time.ZonedDateTime

class OfflineModeTest {
    private val now = ZonedDateTime.of(2026, 10, 29, 14, 5, 0, 0, ZoneId.of("Europe/London"))

    @Test fun phoneAnswersTimeDateAndBatteryItself() {
        assertEquals("It's 14:05.", OfflineAnswers.answerFor("What time is it?", now) { 80 })
        assertEquals("It's 14:05.", OfflineAnswers.answerFor("ORION, what's the time", now) { 80 })
        assertEquals("It's Thursday 29 October 2026.", OfflineAnswers.answerFor("what day is it", now) { 80 })
        assertEquals("The battery is at 80%.", OfflineAnswers.answerFor("battery", now) { 80 })
        assertEquals("I can't read the battery right now.", OfflineAnswers.answerFor("how much battery is left", now) { null })
    }

    @Test fun anythingElseWaitsForTheHostedAi() {
        assertNull(OfflineAnswers.answerFor("what time is the meeting tomorrow", now) { 80 })
        assertNull(OfflineAnswers.answerFor("text Mum that I'm late", now) { 80 })
        assertNull(OfflineAnswers.answerFor("", now) { 80 })
    }

    @Test fun queuedRequestsAreBoundedOrderedAndExpire() {
        var clock = 0L
        val queue = OfflineQueue { clock }
        for (i in 1..OfflineQueue.CAPACITY) assertTrue(queue.add("request $i"))
        assertFalse(queue.add("one too many"))                  // refused, nothing evicted
        assertFalse(queue.add("   "))
        assertEquals(OfflineQueue.CAPACITY, queue.size)
        assertEquals("request 1", queue.poll())
        clock += OfflineQueue.MAX_AGE_MS + 1                    // asked too long ago
        assertEquals(0, queue.size)
        assertNull(queue.poll())
        assertTrue(queue.add("fresh"))
        queue.clear()
        assertFalse(queue.isNotEmpty())
    }

    @Test fun cachedFaceStyleAcceptsOnlyKnownDesignsAndPlainColours() {
        val style = FaceStyle.parse(JSONObject("""{"type":"face","design":"Arc","primary":"#FF0000","accent":"red","background":"#000000"}"""))
        assertEquals(FaceStyle("arc", "#ff0000", FaceStyle.DEFAULT.accent, "#000000"), style)
        assertNull(FaceStyle.parse(JSONObject("""{"design":"<script>"}""")))
        assertNull(FaceStyle.parse(null))
    }

    @Test fun lanRediscoveryScansTheOwnSubnetNearestFirst() {
        val hosts = LanDiscovery.subnetHosts(byteArrayOf(192.toByte(), 168.toByte(), 1, 20), 24)
        assertEquals(253, hosts.size)                            // .1-.254 without the phone itself
        assertFalse("192.168.1.20" in hosts)
        assertFalse("192.168.1.0" in hosts || "192.168.1.255" in hosts)
        assertEquals(setOf("192.168.1.19", "192.168.1.21"), hosts.take(2).toSet())
        // A wide network is searched as the phone's own /24, never 65 000 hosts.
        assertEquals(253, LanDiscovery.subnetHosts(byteArrayOf(10, 0, 5, 9), 16).size)
        assertEquals(1021, LanDiscovery.subnetHosts(byteArrayOf(10, 0, 5, 9), 22).size)
    }

    // ── Calendar, Focus and Fitness on the phone (3.18) ─────────────────────

    @Test fun aTaskWrittenAsOneLineBecomesItsSteps() {
        assertEquals(listOf("Outline the essay", "Write the intro", "Add two sources"),
            DayPlanner.chunkTask("outline the essay, write the intro then add two sources"))
        assertEquals(listOf("Read chapter 3", "Make notes", "Test myself on them"),
            DayPlanner.chunkTask("1. read chapter 3\n2. make notes and test myself on them"))
        assertEquals(listOf("Buy salt and pepper"), DayPlanner.chunkTask("buy salt and pepper"))
        assertTrue(DayPlanner.chunkTask("  ").isEmpty())
        assertEquals(8, DayPlanner.chunkTask((0 until 20).joinToString(", ") { "step $it" }).size)
    }

    @Test fun theFocusDialReadsIdleRunningDueAndBreak() {
        val start = now.toInstant().toEpochMilli()
        assertEquals("25:00", DayPlanner.dial(null, start, 25).text)
        val block = DayPlanner.Block("Write up", start, 20, DayPlanner.breakFor(20), bodyDouble = false)
        val running = DayPlanner.dial(block, start + 5 * 60_000, 25)
        assertEquals("running", running.mode)
        assertEquals("15:00", running.text)
        assertEquals(0.25f, running.progress, 0.001f)
        assertEquals("due", DayPlanner.dial(block, start + 21 * 60_000, 25).mode)
        val rest = DayPlanner.dial(null, start, 25, breakUntil = start + 120_000, breakMs = 300_000)
        assertEquals("break", rest.mode)
        assertEquals("02:00", rest.text)
        assertEquals(4, DayPlanner.breakFor(20))
        assertEquals(3, DayPlanner.breakFor(10))
        // Stopping past halfway still counts the block.
        assertTrue(DayPlanner.gentleStopCounts(block, start + 11 * 60_000))
        assertFalse(DayPlanner.gentleStopCounts(block, start + 4 * 60_000))
        assertEquals(block, DayPlanner.Block.fromJson(block.toJson()))
    }

    @Test fun focusMinutesBlocksAndStreakAreCountedInLocalDays() {
        val today = now.toInstant().toEpochMilli()
        val day = 86_400_000L
        val history = listOf(today - 3 * day to 25, today - day to 45, today - 60_000 to 25, today to 15)
        val focus = DayPlanner.focusDay(history, now)
        assertEquals(40, focus.minutes)
        assertEquals(2, focus.blocks)
        assertEquals(2, focus.streak)
    }

    @Test fun workoutsMakeTheWeekTheStreakAndTheirOwnWords() {
        val at = now.toInstant().toEpochMilli()                     // a Thursday
        val day = 86_400_000L
        val workouts = listOf(
            DayPlanner.Workout(1, DayPlanner.normaliseKind("jogging"), 30.0, 5.0, at - day),
            DayPlanner.Workout(2, DayPlanner.normaliseKind("gym"), 45.0, null, at),
            DayPlanner.Workout(3, "cycle", 61.0, 20.5, at - 9 * day))  // last week
        val week = DayPlanner.week(workouts, now, target = 4)
        assertEquals(2, week.workouts)
        assertEquals(75, week.minutes)
        assertEquals(5.0, week.distanceKm, 0.001)
        assertEquals(2, week.streak)
        assertEquals(4, week.target)
        assertEquals(java.time.DayOfWeek.MONDAY, week.start.dayOfWeek)
        assertEquals(45.0, week.days.first { it.first == now.toLocalDate() }.second, 0.001)
        assertEquals("1:15", DayPlanner.clock(75.0))
        assertEquals("45m", DayPlanner.clock(45.0))
        assertEquals("30 min · 5 km", DayPlanner.amountText(workouts[0]))
        assertEquals("Yesterday", DayPlanner.whenText(at - day, now))
        assertEquals("A workout lasts between a minute and a day.", DayPlanner.checkWorkout(0.0, null))
        assertEquals("That distance does not look right.", DayPlanner.checkWorkout(30.0, 5000.0))
        assertNull(DayPlanner.checkWorkout(30.0, null))
        assertEquals(workouts[0], DayPlanner.Workout.fromJson(JSONObject(workouts[0].toJson().toString())))
        assertEquals("other", DayPlanner.normaliseKind("juggling"))
    }

    @Test fun theCalendarPutsTheNextThingFirstAndFilesTheWeekByDay() {
        val today = now.toLocalDate()
        val entries = listOf(
            DayPlanner.Entry(today, "Lecture", now.withHour(15).withMinute(0), "University"),
            DayPlanner.Entry(today, "Call the dentist", now.plusMinutes(40), "Personal"),
            DayPlanner.Entry(today.plusDays(2), "Open day", null, "University"),
            DayPlanner.Entry(today, "Earlier", now.minusHours(2), "Personal"))
        val next = DayPlanner.nextUp(entries, now)
        assertEquals("Call the dentist", next?.title)
        assertEquals("in 40 min", DayPlanner.relative(next!!, now))
        assertEquals("Saturday", DayPlanner.relative(entries[2], now))
        val week = DayPlanner.weekOf(entries, today)
        assertEquals(7, week.size)
        assertEquals(listOf("Earlier", "Call the dentist", "Lecture"), week[0].second.map { it.title })
        assertEquals("All day", week[2].second.single().timeText())
        assertNull(DayPlanner.nextUp(emptyList(), now))
    }

    @Test fun aShortAskOpensItsPageAndAnythingElseGoesToTheAi() {
        assertEquals("calendar", DayPlanner.pageFor("Open my calendar"))
        assertEquals("calendar", DayPlanner.pageFor("ORION, show me the week ahead."))
        assertEquals("focus", DayPlanner.pageFor("start a focus block"))
        assertEquals("focus", DayPlanner.pageFor("body doubling"))
        assertEquals("fitness", DayPlanner.pageFor("log a workout"))
        assertEquals("fitness", DayPlanner.pageFor("my workouts"))
        assertNull(DayPlanner.pageFor("what's on my calendar tomorrow"))
        assertNull(DayPlanner.pageFor("how do I focus better when studying"))
        assertNull(DayPlanner.pageFor("text Mum that I'm late"))
    }
}
