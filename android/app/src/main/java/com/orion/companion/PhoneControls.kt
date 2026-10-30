package com.orion.companion

import android.app.Activity
import android.content.Intent
import android.text.InputType
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Toast
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Locale

/** Manual phone actions share the same reviewed execution path as voice commands. */
object PhoneControls {
    private data class Field(val key: String, val label: String, val type: Int = InputType.TYPE_CLASS_TEXT,
        val multiline: Boolean = false)

    fun show(activity: Activity, propose: (JSONObject) -> Unit) {
        val choices = arrayOf("Open phone app", "Draft e-mail", "Send reviewed Gmail e-mail", "Call contact or number",
            "Text by SMS", "Open web link", "Show place on map", "Open navigation", "Share text", "Set alarm",
            "Set timer", "Create contact draft", "Create calendar draft", "Find exact contact", "Phone settings",
            "Back", "Home", "Popular YouTube videos", "Phone battery & connection", "Installed apps")
        MaterialAlertDialogBuilder(activity).setTitle("Phone controls").setItems(choices) { _, index ->
            when (index) {
                0 -> chooseApp(activity, propose)
                1, 2 -> form(activity, choices[index], if (index == 1) "compose_email" else "send_email",
                    listOf(Field("to", "Exact e-mail address or contact name"), Field("subject", "Subject"),
                        Field("body", "Message", multiline = true)), propose)
                3 -> form(activity, choices[index], "make_call", listOf(Field("number", "Exact contact name or phone number")), propose)
                4 -> form(activity, choices[index], "send_sms", listOf(Field("number", "Exact contact name or phone number"),
                    Field("body", "Message", multiline = true)), propose)
                5 -> form(activity, choices[index], "open_url", listOf(Field("url", "https://example.org", InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI)), propose)
                6, 7 -> form(activity, choices[index], if (index == 6) "map" else "navigate",
                    listOf(Field("query", "Place or address")), propose)
                8 -> form(activity, choices[index], "share", listOf(Field("text", "Text to share", multiline = true)), propose)
                9 -> form(activity, choices[index], "set_alarm", listOf(Field("hour", "Hour (0–23)", InputType.TYPE_CLASS_NUMBER),
                    Field("minutes", "Minutes (0–59)", InputType.TYPE_CLASS_NUMBER), Field("label", "Label (optional)")), propose)
                10 -> form(activity, choices[index], "set_timer", listOf(Field("seconds", "Duration in seconds", InputType.TYPE_CLASS_NUMBER),
                    Field("label", "Label (optional)")), propose)
                11 -> form(activity, choices[index], "create_contact", listOf(Field("name", "Contact name"),
                    Field("number", "Number (optional)", InputType.TYPE_CLASS_PHONE), Field("email", "E-mail (optional)")), propose)
                12 -> form(activity, choices[index], "create_calendar_event", listOf(Field("title", "Title"),
                    Field("start", "Start: yyyy-MM-dd HH:mm"), Field("end", "End: yyyy-MM-dd HH:mm"),
                    Field("location", "Location (optional)"), Field("description", "Description (optional)", multiline = true)), propose)
                13 -> form(activity, choices[index], "find_contact", listOf(Field("name", "Exact contact name")), propose)
                14 -> {
                    val screens = arrayOf("app", "accessibility", "wifi", "bluetooth", "notifications", "location", "sound", "display", "privacy", "battery")
                    MaterialAlertDialogBuilder(activity).setTitle("Phone settings").setItems(screens) { _, chosen ->
                        propose(JSONObject().put("kind", "open_settings").put("screen", screens[chosen]))
                    }.setNegativeButton("Cancel", null).show()
                }
                15, 16 -> propose(JSONObject().put("kind", if (index == 15) "back" else "home"))
                17 -> propose(JSONObject().put("kind", "show_trending_videos"))
                18 -> propose(JSONObject().put("kind", "device_status"))
                19 -> propose(JSONObject().put("kind", "list_apps"))
            }
        }.setNegativeButton("Close", null).show()
    }

    private fun chooseApp(activity: Activity, propose: (JSONObject) -> Unit) {
        @Suppress("DEPRECATION")
        val apps = try {
            activity.packageManager.queryIntentActivities(
                Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)
                .map { it.loadLabel(activity.packageManager).toString() to it.activityInfo.packageName }
                .distinctBy { it.second }.sortedBy { it.first.lowercase(Locale.ROOT) }
        } catch (_: Exception) {
            Toast.makeText(activity, "Phone apps could not be listed", Toast.LENGTH_LONG).show()
            return
        }
        if (apps.isEmpty()) { Toast.makeText(activity, "No launchable phone apps are visible", Toast.LENGTH_LONG).show(); return }
        val labels = apps.map { "${it.first}\n${it.second}" }.toTypedArray()
        MaterialAlertDialogBuilder(activity).setTitle("Open phone app").setItems(labels) { _, index ->
            propose(JSONObject().put("kind", "open_app").put("package", apps[index].second))
        }.setNegativeButton("Cancel", null).show()
    }

    private fun form(activity: Activity, title: String, kind: String, fields: List<Field>,
        propose: (JSONObject) -> Unit) {
        val density = activity.resources.displayMetrics.density
        val pad = (20 * density).toInt()
        val layout = LinearLayout(activity).apply { orientation = LinearLayout.VERTICAL; setPadding(pad, 0, pad, 0) }
        val inputs = fields.associateWith { field ->
            EditText(activity).apply {
                hint = field.label
                inputType = field.type or if (field.multiline) InputType.TYPE_TEXT_FLAG_MULTI_LINE else 0
                if (field.multiline) minLines = 4
                layout.addView(this)
            }
        }
        val dialog = MaterialAlertDialogBuilder(activity).setTitle(title)
            .setView(ScrollView(activity).apply { addView(layout) })
            .setPositiveButton("Review", null).setNegativeButton("Cancel", null).create()
        dialog.setOnShowListener {
            dialog.getButton(android.content.DialogInterface.BUTTON_POSITIVE).setOnClickListener {
                val action = JSONObject().put("kind", kind)
                var valid = true
                inputs.forEach { (field, input) ->
                    val value = input.text.toString()
                    when (field.key) {
                        "hour", "minutes", "seconds" -> {
                            val number = value.toLongOrNull()
                            if (number == null) { input.error = "Enter a whole number"; valid = false }
                            else action.put(field.key, number)
                        }
                        "start", "end" -> {
                            val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.UK).apply { isLenient = false }
                            val parsed = runCatching { formatter.parse(value) }.getOrNull()
                            if (parsed == null || formatter.format(parsed) != value) {
                                input.error = "Use yyyy-MM-dd HH:mm"; valid = false
                            } else action.put(field.key, parsed.time)
                        }
                        else -> action.put(field.key, value)
                    }
                }
                if (valid) { dialog.dismiss(); propose(action) }
            }
        }
        dialog.show()
    }
}
