package com.orion.companion

/** Explicit methods accepted by the native phone; the network cannot name arbitrary intents. */
internal object PhoneCommandContract {
    val deviceActions = setOf("open_app", "list_apps", "device_status", "find_contact", "make_call", "send_sms",
        "compose_email", "send_email", "open_url", "map", "navigate", "share", "open_settings",
        "set_alarm", "set_timer", "create_contact", "create_calendar_event", "show_trending_videos",
        "back", "home")
    val deviceMethods = deviceActions.map { "device.$it" }.toSet()
    val methods = deviceMethods + setOf("viewport.capture_dom", "viewport.click", "viewport.scroll", "viewport.set_text")
}
