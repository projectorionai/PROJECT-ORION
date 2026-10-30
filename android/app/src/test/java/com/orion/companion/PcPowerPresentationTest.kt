package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class PcPowerPresentationTest {
    @Test fun onlyTheCountdownCanBeCancelled() {
        assertTrue(PcPowerPresentation.canCancel("scheduled"))
        listOf("executing", "initiated", "blocked", "uncertain", "cancelled", "")
            .forEach { assertFalse(PcPowerPresentation.canCancel(it)) }
    }
    @Test fun anActiveShutdownCannotBeScheduledAgain() {
        assertFalse(PcPowerPresentation.canSchedule("scheduled"))
        assertFalse(PcPowerPresentation.canSchedule("executing"))
        assertTrue(PcPowerPresentation.canSchedule("cancelled"))
    }
    @Test fun acknowledgementIsNotPresentedAsProvenPowerOff() {
        assertEquals("Windows accepted shutdown", PcPowerPresentation.label("initiated", 0))
        assertEquals("Shutdown outcome unknown", PcPowerPresentation.label("uncertain", 0))
        assertEquals("Shutdown in 0 seconds", PcPowerPresentation.label("scheduled", -5))
    }
}
