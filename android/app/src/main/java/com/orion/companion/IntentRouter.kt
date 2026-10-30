package com.orion.companion

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.content.pm.PackageManager
import android.net.Uri
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.os.PowerManager
import android.provider.AlarmClock
import android.provider.CalendarContract
import android.provider.ContactsContract
import android.provider.Settings
import android.telephony.SmsManager
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject

/** Reviewed native phone actions. Receipts describe what Android actually accepted. */
class IntentRouter(private val activity: Activity) {
    fun route(action: JSONObject, authorize: (String, () -> Unit) -> Unit,
        requestPermission: (String, (Boolean) -> Unit) -> Unit, report: (JSONObject) -> Unit,
        canExecute: () -> Boolean = { true },
        allowHandsFreeLaunch: Boolean = false) {
        val kind = action.optString("kind")
        fun reviewed(details: String, execute: () -> Unit) {
            if (!canExecute()) { report(result(false, "Command expired or disconnected", "expired")); return }
            authorize(details) {
                if (canExecute()) execute() else report(result(false, "Command expired or disconnected", "expired"))
            }
        }
        fun launch(intent: Intent, detail: String, status: String = "opened") {
            if (!canExecute()) { report(result(false, "Command expired or disconnected", "expired")); return }
            try {
                activity.startActivity(intent)
                report(result(true, detail, status).put("action", kind))
            } catch (_: Exception) {
                report(result(false, "No available phone application could perform this action", "unavailable"))
            }
        }
        when (kind) {
            "open_app" -> {
                val suppliedPackage = action.optString("package")
                val suppliedName = action.optString("name")
                if (suppliedPackage.isBlank() && suppliedName.isBlank()) {
                    report(result(false, "Which app would you like me to open?", "name_needed")); return
                }
                val match = PhoneActionRules.resolveApp(suppliedName, suppliedPackage, installedApps())
                val selected = match.app
                if (selected == null) {
                    if (match.choices.isNotEmpty()) {
                        val choices = match.choices.take(8)
                        val names = choices.joinToString(", ") { it.label }
                        val receipt = result(false, "I found several apps: $names. Which one would you like?", "ambiguous")
                            .put("choices", JSONArray().apply {
                                choices.forEach { put(JSONObject().put("name", it.label).put("package", it.packageName)) }
                            })
                        report(receipt)
                    } else report(result(false, "I couldn't find a matching installed app. Ask me to list your apps if you like.", "unavailable"))
                    return
                }
                val openResolvedApp = {
                    val intent = activity.packageManager.getLaunchIntentForPackage(selected.packageName)
                    if (intent == null) report(result(false, "App is not installed or visible", "unavailable"))
                    else launch(intent, "${selected.label} opened on the phone", "app_opened")
                }
                if (allowHandsFreeLaunch && PhoneActionRules.mayLaunchWithoutReview(kind, selected.packageName))
                    openResolvedApp()
                else reviewed("PHONE APP: ${selected.label}\nACTION: Open application", openResolvedApp)
            }
            "list_apps" -> reviewed("PHONE APPS\nACTION: Share installed application names with ORION") {
                val apps = JSONArray()
                installedApps().take(300).forEach { app ->
                    apps.put(JSONObject().put("name", app.label).put("package", app.packageName)
                        .put("isGame", app.isGame))
                }
                report(result(true, "Installed phone apps read", "read").put("apps", apps))
            }
            "device_status" -> reviewed("PHONE STATUS\nACTION: Share battery, charging and connection state with ORION") {
                val receipt = runCatching {
                    val battery = activity.getSystemService(BatteryManager::class.java)
                        ?.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY) ?: -1
                    val state = activity.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                        ?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
                    val charging = state == BatteryManager.BATTERY_STATUS_CHARGING ||
                        state == BatteryManager.BATTERY_STATUS_FULL
                    val connectivity = activity.getSystemService(ConnectivityManager::class.java)
                    val network = connectivity?.activeNetwork?.let(connectivity::getNetworkCapabilities)
                    val connection = when {
                        network == null -> "offline"
                        network.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "Wi-Fi"
                        network.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "mobile data"
                        network.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "Ethernet"
                        else -> "connected"
                    }
                    val powerSaving = activity.getSystemService(PowerManager::class.java)?.isPowerSaveMode ?: false
                    val detail = "Battery ${if (battery in 0..100) "$battery%" else "unavailable"}, " +
                        "${if (charging) "charging" else "not charging"}; $connection; " +
                        "power saving ${if (powerSaving) "on" else "off"}."
                    result(true, detail, "read").put("battery_percent", if (battery in 0..100) battery else JSONObject.NULL)
                        .put("charging", charging).put("connection", connection).put("power_saving", powerSaving)
                }.getOrElse { result(false, "Phone status could not be read", "unavailable") }
                report(receipt)
            }
            "find_contact" -> {
                val name = PhoneActionRules.contactName(action.optString("name"))
                if (name == null) { report(result(false, "Specify an exact contact name")); return }
                reviewed("PHONE CONTACT: $name\nACTION: Share only this contact's numbers and e-mail addresses") {
                    withPermission(Manifest.permission.READ_CONTACTS, requestPermission, report) {
                        lookupContact(name, canExecute) { report(it ?: result(false, "Contact not found or ambiguous", "unavailable")) }
                    }
                }
            }
            "make_call", "call" -> resolveContactValue(action.optString("number"), false, requestPermission, report) { number ->
                reviewed("PHONE CALL: $number\nACTION: Start call") {
                    withPermission(Manifest.permission.CALL_PHONE, requestPermission, report) {
                        launch(Intent(Intent.ACTION_CALL, Uri.fromParts("tel", number, null)),
                            "Call requested on the phone; connection is not yet confirmed", "call_started")
                    }
                }
            }
            "send_sms", "sms" -> {
                val body = action.optString("body")
                if (body.isBlank() || body.length > 1600) { report(result(false, "SMS body must contain 1–1,600 characters")); return }
                resolveContactValue(action.optString("number"), false, requestPermission, report) { number ->
                    val directSms = ContextCompat.checkSelfPermission(activity, Manifest.permission.SEND_SMS) ==
                        PackageManager.PERMISSION_GRANTED
                    reviewed("PHONE SMS TO: $number\nMESSAGE:\n$body\n\nACTION: ${if (directSms) "Submit through the mobile network" else "Open a reviewed Messages draft for you to send"}") {
                        if (!canExecute()) { report(result(false, "Command expired or disconnected")); return@reviewed }
                        if (directSms && ContextCompat.checkSelfPermission(activity, Manifest.permission.SEND_SMS) ==
                            PackageManager.PERMISSION_GRANTED) {
                            try {
                                @Suppress("DEPRECATION") val sms = SmsManager.getDefault()
                                val parts = sms.divideMessage(body)
                                if (parts.size == 1) sms.sendTextMessage(number, null, body, null, null)
                                else sms.sendMultipartTextMessage(number, null, parts, null, null)
                                // SmsManager queues the operation asynchronously. This
                                // call cannot establish that a carrier accepted it.
                                report(result(true, "Android accepted the SMS submission request; network acceptance and delivery are not confirmed", "submission_requested"))
                            } catch (_: Exception) { report(result(false, "Unable to submit SMS", "failed")) }
                        } else {
                            // SEND_SMS is installer-restricted on recent Android. The
                            // normal messaging app can still compose without it.
                            val draft = Intent(Intent.ACTION_SENDTO, Uri.fromParts("smsto", number, null))
                                .putExtra("sms_body", body).putExtra(Intent.EXTRA_TEXT, body)
                            launch(draft, "Reviewed SMS draft opened in the phone's messaging app; tap Send there", "draft_opened")
                        }
                    }
                }
            }
            "compose_email", "email", "send_email" -> {
                val subject = action.optString("subject")
                val body = action.optString("body")
                if (subject.length > 200 || subject.any { it == '\r' || it == '\n' } || body.length > 16_000) {
                    report(result(false, "Invalid e-mail subject or body")); return
                }
                val send = kind == "send_email"
                val requestedPackage = action.optString("package")
                if (requestedPackage.isNotBlank() && !PhoneActionRules.packagePattern.matches(requestedPackage)) {
                    report(result(false, "Invalid e-mail app package")); return
                }
                if (send && requestedPackage.isNotBlank() && requestedPackage != PhoneActionRules.GMAIL_PACKAGE) {
                    report(result(false, "Verified sending supports Gmail only; use compose_email for another client", "unsupported_client")); return
                }
                if (send && !OrionAccessibilityService.isConnected()) {
                    report(result(false, "Enable Phone accessibility before verified Gmail sending; no e-mail was sent", "accessibility_required")); return
                }
                resolveContactValue(action.optString("to"), true, requestPermission, report) { address ->
                    reviewed("PHONE E-MAIL TO: $address\nSUBJECT: $subject\nBODY:\n$body\nACTION: ${if (send) "Send this exact message from Gmail" else "Open reviewed draft; do not send"}") {
                        val selectedPackage = if (send) PhoneActionRules.GMAIL_PACKAGE else requestedPackage
                        val mail = Intent(Intent.ACTION_SENDTO,
                            Uri.parse("mailto:${Uri.encode(address)}?subject=${Uri.encode(subject)}&body=${Uri.encode(body)}"))
                            .putExtra(Intent.EXTRA_EMAIL, arrayOf(address))
                            .putExtra(Intent.EXTRA_SUBJECT, subject).putExtra(Intent.EXTRA_TEXT, body)
                        if (selectedPackage.isNotBlank()) mail.setPackage(selectedPackage)
                        if (!send) launch(mail, "Reviewed phone e-mail draft opened; it has not been sent", "draft_opened")
                        else try {
                            activity.startActivity(mail)
                            OrionAccessibilityService.sendReviewedEmail(
                                PhoneActionRules.EmailRequest(address, subject, body), canExecute, report)
                        } catch (_: Exception) { report(result(false, "Gmail is unavailable; no e-mail was sent", "unavailable")) }
                    }
                }
            }
            "open_url" -> {
                val url = PhoneActionRules.webUrl(action.optString("url"))
                if (url == null) { report(result(false, "Specify an HTTP or HTTPS link without credentials")); return }
                reviewed("PHONE WEB LINK: $url\nACTION: Open in the phone's browser") {
                    launch(Intent(Intent.ACTION_VIEW, Uri.parse(url)), "Web link opened on the phone", "link_opened")
                }
            }
            "navigate", "map" -> {
                val query = action.optString("query").trim()
                if (query.length !in 1..1000 || query.any { it.isISOControl() }) { report(result(false, "Specify a place or address")); return }
                val coordinates = Regex("^([+-]?\\d{1,2}(?:\\.\\d+)?),\\s*([+-]?\\d{1,3}(?:\\.\\d+)?)$")
                    .matchEntire(query)
                val validCoordinates = coordinates != null &&
                    coordinates.groupValues[1].toDoubleOrNull()?.let { it in -90.0..90.0 } == true &&
                    coordinates.groupValues[2].toDoubleOrNull()?.let { it in -180.0..180.0 } == true
                if (validCoordinates) {
                    val openBuiltInMap = {
                        launch(Intent(activity, OfflineMapActivity::class.java)
                            .putExtra("coordinate_query", query)
                            .putExtra("route_to_coordinate", kind == "navigate"),
                            if (kind == "navigate") "ORION's route screen opened. Offline calculation needs BRouter, downloaded road data and a fresh location; no route is confirmed yet."
                            else "ORION's offline map screen opened at the requested coordinates; map coverage depends on installed packs",
                            "offline_map_opened")
                    }
                    if (allowHandsFreeLaunch) openBuiltInMap()
                    else reviewed("ORION OFFLINE MAP: $query\nACTION: ${if (kind == "navigate") "Calculate a car route" else "Show coordinates"} in the phone map", openBuiltInMap)
                    return
                }
                val offline = offlineMapsPackage()
                if (offline == null) {
                    report(result(false, "Install OsmAnd and download every map for your journey in OsmAnd. Open Options → Phone → Offline UK map for ORION's position map. No route was started.", "offline_maps_needed"))
                    return
                }
                val openOfflineMap = {
                    val direct = kind == "navigate" && validCoordinates
                    val uri = if (direct) "google.navigation:q=${Uri.encode(query)}"
                        else "geo:0,0?q=${Uri.encode(query)}"
                    launch(Intent(Intent.ACTION_VIEW, Uri.parse(uri)).setPackage(offline),
                        if (kind == "navigate" && direct) "OsmAnd opened the coordinate route; check it and the downloaded maps before driving" else if (kind == "navigate")
                            "OsmAnd opened the offline destination search; choose the exact result and start the Car route" else
                            "Place opened in OsmAnd using its on-phone map data",
                        if (kind == "navigate" && direct) "offline_navigation_opened" else "offline_map_opened")
                }
                if (allowHandsFreeLaunch && PhoneActionRules.mayLaunchWithoutReview(kind, offline))
                    openOfflineMap()
                else reviewed("OFFLINE PHONE MAP: $query\nACTION: ${if (kind == "navigate") "Prepare car navigation" else "Show place"} in OsmAnd", openOfflineMap)
            }
            "show_trending_videos" -> reviewed("YOUTUBE CHARTS\nACTION: Open popular videos for Great Britain") {
                launch(Intent(Intent.ACTION_VIEW,
                    Uri.parse("https://charts.youtube.com/charts/TrendingVideos/gb/weekly")),
                    "YouTube's current popular-video chart opened on the phone", "chart_opened")
            }
            "share" -> {
                val text = action.optString("text")
                if (text.isBlank() || text.length > 16_000) { report(result(false, "Specify 1–16,000 characters to share")); return }
                reviewed("PHONE SHARE SHEET\nCONTENT:\n$text\nACTION: Open share sheet; choose the recipient yourself") {
                    launch(Intent.createChooser(Intent(Intent.ACTION_SEND).setType("text/plain")
                        .putExtra(Intent.EXTRA_TEXT, text), "Share from ORION"),
                        "Phone share sheet opened; nothing has been sent", "share_sheet_opened")
                }
            }
            "open_settings" -> {
                val screen = action.optString("screen")
                val settingsAction = SETTINGS[screen]
                if (settingsAction == null) { report(result(false, "Unsupported phone settings screen")); return }
                reviewed("PHONE SETTINGS: $screen\nACTION: Open settings") {
                    val intent = Intent(settingsAction)
                    if (screen == "app") intent.data = Uri.parse("package:${activity.packageName}")
                    if (screen == "notifications") intent.putExtra(Settings.EXTRA_APP_PACKAGE, activity.packageName)
                    launch(intent, "Phone settings opened; no setting was changed automatically", "settings_opened")
                }
            }
            "set_alarm" -> {
                val hour = exactLong(action, "hour")
                val minutes = exactLong(action, "minutes")
                val label = action.optString("label")
                if (hour == null || hour !in 0L..23L || minutes == null || minutes !in 0L..59L || label.length > 200) {
                    report(result(false, "Specify an alarm hour 0–23 and minutes 0–59")); return
                }
                reviewed("PHONE ALARM: %02d:%02d\nLABEL: %s\nACTION: Open clock to review the alarm".format(hour, minutes, label)) {
                    launch(Intent(AlarmClock.ACTION_SET_ALARM).putExtra(AlarmClock.EXTRA_HOUR, hour.toInt())
                        .putExtra(AlarmClock.EXTRA_MINUTES, minutes.toInt()).putExtra(AlarmClock.EXTRA_MESSAGE, label)
                        .putExtra(AlarmClock.EXTRA_SKIP_UI, false), "Phone clock opened to review the alarm", "clock_opened")
                }
            }
            "set_timer" -> {
                val seconds = exactLong(action, "seconds")
                val label = action.optString("label")
                if (seconds == null || seconds !in 1L..86_400L || label.length > 200) { report(result(false, "Specify a timer of 1–86,400 seconds")); return }
                reviewed("PHONE TIMER: $seconds seconds\nLABEL: $label\nACTION: Open clock to review the timer") {
                    launch(Intent(AlarmClock.ACTION_SET_TIMER).putExtra(AlarmClock.EXTRA_LENGTH, seconds.toInt())
                        .putExtra(AlarmClock.EXTRA_MESSAGE, label).putExtra(AlarmClock.EXTRA_SKIP_UI, false),
                        "Phone clock opened to review the timer", "clock_opened")
                }
            }
            "create_contact" -> {
                val name = PhoneActionRules.contactName(action.optString("name"))
                val number = action.optString("number").takeIf { it.isNotBlank() }?.let(PhoneActionRules::number)
                val email = action.optString("email").takeIf { it.isNotBlank() }?.let(PhoneActionRules::email)
                if (name == null || (number == null && email == null) ||
                    (action.optString("number").isNotBlank() && number == null) ||
                    (action.optString("email").isNotBlank() && email == null)) {
                    report(result(false, "Specify an exact contact name and a valid number or e-mail address")); return
                }
                reviewed("PHONE CONTACT: $name\nNUMBER: ${number.orEmpty()}\nE-MAIL: ${email.orEmpty()}\nACTION: Open contact draft") {
                    launch(Intent(Intent.ACTION_INSERT, ContactsContract.Contacts.CONTENT_URI)
                        .putExtra(ContactsContract.Intents.Insert.NAME, name)
                        .putExtra(ContactsContract.Intents.Insert.PHONE, number)
                        .putExtra(ContactsContract.Intents.Insert.EMAIL, email),
                        "Phone contact draft opened; save it in Contacts", "draft_opened")
                }
            }
            "create_calendar_event" -> {
                val title = action.optString("title")
                val start = exactLong(action, "start")
                val end = exactLong(action, "end")
                val location = action.optString("location")
                val description = action.optString("description")
                if (title.isBlank() || title.length > 200 || start == null || start !in 1L..253_402_300_799_000L ||
                    end == null || end !in 1L..253_402_300_799_000L || end <= start ||
                    location.length > 1000 || description.length > 4000) {
                    report(result(false, "Specify an event title and an end time after its start")); return
                }
                reviewed("PHONE CALENDAR: $title\nSTART: ${java.util.Date(start)}\nEND: ${java.util.Date(end)}\nPLACE: $location\nDESCRIPTION:\n$description\nACTION: Open event draft") {
                    launch(Intent(Intent.ACTION_INSERT, CalendarContract.Events.CONTENT_URI)
                        .putExtra(CalendarContract.Events.TITLE, title)
                        .putExtra(CalendarContract.EXTRA_EVENT_BEGIN_TIME, start)
                        .putExtra(CalendarContract.EXTRA_EVENT_END_TIME, end)
                        .putExtra(CalendarContract.Events.EVENT_LOCATION, location)
                        .putExtra(CalendarContract.Events.DESCRIPTION, description),
                        "Phone calendar draft opened; save it in Calendar", "draft_opened")
                }
            }
            "back", "home" -> reviewed("PHONE NAVIGATION\nACTION: ${if (kind == "back") "Go back" else "Go to Home"}") {
                OrionAccessibilityService.navigate(kind, report)
            }
            else -> report(result(false, "Unsupported phone action", "unsupported"))
        }
    }

    private fun installedApps(): List<PhoneActionRules.App> = try {
        @Suppress("DEPRECATION")
        activity.packageManager.queryIntentActivities(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
            .map { info ->
                val application = info.activityInfo.applicationInfo
                PhoneActionRules.App(info.loadLabel(activity.packageManager).toString().take(80),
                    info.activityInfo.packageName,
                    application.category == ApplicationInfo.CATEGORY_GAME ||
                        application.flags and ApplicationInfo.FLAG_IS_GAME != 0)
            }
            .distinctBy { it.packageName }.sortedBy { it.label.lowercase() }
    } catch (_: Exception) { emptyList() }

    private fun offlineMapsPackage(): String? = PhoneActionRules.offlineMapsPackages
        .firstOrNull { activity.packageManager.getLaunchIntentForPackage(it) != null }

    private fun withPermission(permission: String, requestPermission: (String, (Boolean) -> Unit) -> Unit,
        report: (JSONObject) -> Unit, execute: () -> Unit) {
        if (ContextCompat.checkSelfPermission(activity, permission) == PackageManager.PERMISSION_GRANTED) execute()
        else requestPermission(permission) { granted ->
            if (granted) execute() else report(result(false, "Phone permission denied", "permission_denied"))
        }
    }

    private fun resolveContactValue(supplied: String, email: Boolean,
        requestPermission: (String, (Boolean) -> Unit) -> Unit, report: (JSONObject) -> Unit, callback: (String) -> Unit) {
        (if (email) PhoneActionRules.email(supplied) else PhoneActionRules.number(supplied))?.let { callback(it); return }
        val name = PhoneActionRules.contactName(supplied)
        if (name == null) { report(result(false, "Specify one exact address, number or contact name")); return }
        withPermission(Manifest.permission.READ_CONTACTS, requestPermission, report) {
            lookupContact(name) { contact ->
                val values = contact?.optJSONArray(if (email) "emails" else "numbers")
                if (values?.length() == 1) callback(values.getString(0))
                else report(result(false, "Contact not found or has several targets; specify one exact address or number"))
            }
        }
    }

    /** Exact names and a single contact ID only; no broad address-book export. */
    private fun lookupContact(name: String, canExecute: () -> Boolean = { true }, callback: (JSONObject?) -> Unit) {
        Thread {
            val ids = linkedSetOf<String>()
            val numbers = linkedSetOf<String>()
            val emails = linkedSetOf<String>()
            var valid = true
            try {
                activity.contentResolver.query(ContactsContract.Contacts.CONTENT_URI,
                    arrayOf(ContactsContract.Contacts._ID), "${ContactsContract.Contacts.DISPLAY_NAME_PRIMARY} = ? COLLATE NOCASE",
                    arrayOf(name), null)?.use { cursor ->
                    while (cursor.moveToNext() && ids.size < 2) ids.add(cursor.getString(0))
                }
                if (ids.size == 1) {
                    val id = ids.single()
                    activity.contentResolver.query(ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                        arrayOf(ContactsContract.CommonDataKinds.Phone.NUMBER),
                        "${ContactsContract.CommonDataKinds.Phone.CONTACT_ID} = ?", arrayOf(id), null)?.use { cursor ->
                        while (cursor.moveToNext() && numbers.size < 10) PhoneActionRules.number(cursor.getString(0).orEmpty())?.let(numbers::add)
                    }
                    activity.contentResolver.query(ContactsContract.CommonDataKinds.Email.CONTENT_URI,
                        arrayOf(ContactsContract.CommonDataKinds.Email.ADDRESS),
                        "${ContactsContract.CommonDataKinds.Email.CONTACT_ID} = ?", arrayOf(id), null)?.use { cursor ->
                        while (cursor.moveToNext() && emails.size < 10) PhoneActionRules.email(cursor.getString(0).orEmpty())?.let(emails::add)
                    }
                }
            } catch (_: Exception) { valid = false }
            activity.runOnUiThread {
                if (!canExecute() || !valid || ids.size != 1) callback(null)
                else callback(result(true, "Exact phone contact read", "read").put("name", name)
                    .put("numbers", JSONArray(numbers.toList())).put("emails", JSONArray(emails.toList())))
            }
        }.apply { isDaemon = true; this.name = "orion-contact-lookup" }.start()
    }

    private fun exactLong(action: JSONObject, key: String): Long? = when (val value = action.opt(key)) {
        is Int -> value.toLong()
        is Long -> value
        else -> null
    }

    private fun result(ok: Boolean, detail: String, status: String = if (ok) "completed" else "failed") =
        JSONObject().put("ok", ok).put("detail", detail).put("status", status)

    companion object {
        private val SETTINGS = mapOf("app" to Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
            "accessibility" to Settings.ACTION_ACCESSIBILITY_SETTINGS, "wifi" to Settings.ACTION_WIFI_SETTINGS,
            "bluetooth" to Settings.ACTION_BLUETOOTH_SETTINGS, "notifications" to Settings.ACTION_APP_NOTIFICATION_SETTINGS,
            "location" to Settings.ACTION_LOCATION_SOURCE_SETTINGS, "sound" to Settings.ACTION_SOUND_SETTINGS,
            "display" to Settings.ACTION_DISPLAY_SETTINGS, "privacy" to Settings.ACTION_PRIVACY_SETTINGS,
            "battery" to Settings.ACTION_BATTERY_SAVER_SETTINGS)
    }
}
