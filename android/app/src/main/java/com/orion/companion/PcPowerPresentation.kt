package com.orion.companion

/** Power labels describe acknowledged stages, without claiming the PC is already off. */
internal object PcPowerPresentation {
    const val assistantScope = "For an explicit PC shutdown request, explain that Options → " +
        "Connections → PC controls provides paired PC shutdown without changing out of phone mode. " +
        "It requires native confirmation and has a 60-second cancellation window. " +
        "The PC must be online with ORION running. The phone_action tools cannot perform PC shutdown; " +
        "do not claim it has been scheduled or completed. Other PC tasks need Desktop mode in Options " +
        "and the PC online. Do not use phone actions to satisfy a request explicitly targeting the PC. "

    fun canCancel(state: String): Boolean = state == "scheduled"
    fun canSchedule(state: String): Boolean = state !in setOf("scheduled", "executing")
    fun label(state: String, seconds: Int): String = when (state) {
        "scheduled" -> "Shutdown in ${seconds.coerceAtLeast(0)} seconds"
        "executing" -> "Requesting Windows shutdown"
        "initiated" -> "Windows accepted shutdown"
        "cancelled" -> "Shutdown cancelled"
        "blocked" -> "Windows did not accept shutdown"
        "uncertain" -> "Shutdown outcome unknown"
        else -> "PC connected"
    }
}
