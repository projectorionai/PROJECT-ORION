package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SentinelLayoutTest {
    @Test fun dragIsBoundedByCutoutAndSystemBars() {
        val bounds = SentinelLayout.Bounds(18, 70, 420, 840)
        assertEquals(SentinelLayout.Position(18, 70), SentinelLayout.clamp(-90, -40, 72, 72, bounds))
        assertEquals(SentinelLayout.Position(348, 768), SentinelLayout.clamp(500, 900, 72, 72, bounds))
    }

    @Test fun menuMovesToLeftAndUpAtBottomRight() {
        val bounds = SentinelLayout.Bounds(6, 60, 394, 760)
        val menu = SentinelLayout.menu(322, 680, 72, 304, 400, 6, bounds)
        assertEquals(SentinelLayout.Position(12, 360), menu)
        assertTrue(menu.x + 304 <= bounds.right)
        assertTrue(menu.y + 400 <= bounds.bottom)
    }

    @Test fun menuRemainsVisibleInLandscapeOrSmallDisplays() {
        val bounds = SentinelLayout.Bounds(30, 20, 690, 290)
        val menu = SentinelLayout.menu(610, 200, 72, 304, 270, 6, bounds)
        assertEquals(SentinelLayout.Position(300, 20), menu)
        assertEquals(SentinelLayout.Position(30, 20), SentinelLayout.clamp(90, 80, 700, 300, bounds))
    }

    @Test fun foregroundContractRejectsUnrecognisedActions() {
        assertTrue(SentinelActionContract.isSupported(SentinelActionContract.DICTATE))
        assertTrue(SentinelActionContract.isSupported(SentinelActionContract.TOGGLE_LISTENING))
        assertFalse(SentinelActionContract.isSupported(null))
        assertFalse(SentinelActionContract.isSupported("send_email"))
    }
}
