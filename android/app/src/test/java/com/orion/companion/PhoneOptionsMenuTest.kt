package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Test

class PhoneOptionsMenuTest {
    @Test fun everyActionHasOnePlaceInTheMenu() {
        val actions = PhoneOptionsMenu.groups.flatMap { it.actions.toList() }
        assertEquals((0..25).toList(), actions.sorted())
    }
}
