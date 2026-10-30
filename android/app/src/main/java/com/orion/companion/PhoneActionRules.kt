package com.orion.companion

import java.net.URI
import java.util.Locale

/** Shared limits and exact matching for reviewed native phone commands. */
internal object PhoneActionRules {
    private val emailPattern = Regex("[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\\.[A-Za-z]{2,63}")
    private val numberPattern = Regex("\\+?[0-9]{3,15}")
    val packagePattern = Regex("[A-Za-z0-9_]+(?:\\.[A-Za-z0-9_]+)+")
    val offlineMapsPackages = setOf("net.osmand", "net.osmand.plus")

    /** Only opening a resolved launcher or OsmAnd may skip the slide. */
    fun mayLaunchWithoutReview(kind: String, targetPackage: String): Boolean = when (kind) {
        "open_app" -> packagePattern.matches(targetPackage)
        "navigate", "map" -> targetPackage in offlineMapsPackages
        else -> false
    }

    fun email(value: String): String? = value.trim().takeIf {
        it.length in 3..254 && !it.contains('\r') && !it.contains('\n') &&
            emailPattern.matches(it) && !it.contains("..")
    }

    fun number(value: String): String? = value.filterNot {
        it == ' ' || it == '-' || it == '(' || it == ')'
    }.takeIf { numberPattern.matches(it) }

    fun contactName(value: String): String? = value.trim().takeIf {
        it.length in 2..80 && it.none { character -> character.isISOControl() }
    }

    fun webUrl(value: String): String? {
        if (value.length !in 1..2048 || value.any { it.isISOControl() }) return null
        val uri = try { URI(value) } catch (_: Exception) { return null }
        if (uri.scheme?.lowercase() !in setOf("https", "http") ||
            uri.host.isNullOrBlank() || uri.rawUserInfo != null) return null
        return value
    }

    /** A label lookup never chooses between several installed applications. */
    fun exactPackage(name: String, apps: List<Pair<String, String>>): String? =
        apps.filter { it.first.equals(name.trim(), ignoreCase = true) }
            .map { it.second }.distinct().singleOrNull()

    data class App(val label: String, val packageName: String, val isGame: Boolean = false)
    data class AppMatch(val app: App? = null, val choices: List<App> = emptyList(),
                        val unavailable: Boolean = false)

    /** Resolve only apps Android reports as launchable; ambiguous requests need a choice. */
    fun resolveApp(name: String, packageName: String, installed: List<App>): AppMatch {
        val apps = installed.filter { packagePattern.matches(it.packageName) }
            .distinctBy { it.packageName }
        val suppliedPackage = packageName.trim()
        val requested = normaliseAppName(name)
        val gameRequest = Regex("^(?:(?:one of )?(?:my )?|some |any |fun )?(?:video )?games?(?: for me)?$")
            .matches(requested) || requested == "something to play"
        if (suppliedPackage.isNotEmpty()) {
            if (!packagePattern.matches(suppliedPackage)) return AppMatch(unavailable = true)
            val exact = apps.firstOrNull { it.packageName == suppliedPackage }
                ?: return AppMatch(unavailable = true)
            if (requested.isNotEmpty() && compactAppName(requested) != compactAppName(exact.label) &&
                !(gameRequest && exact.isGame)) return AppMatch(unavailable = true)
            return AppMatch(app = exact)
        }
        if (requested.isEmpty()) return AppMatch(unavailable = true)
        val matches = if (gameRequest) apps.filter { it.isGame } else {
            val compact = compactAppName(requested)
            val exact = apps.filter { compactAppName(it.label) == compact }
            if (exact.isNotEmpty()) exact else apps.filter { matchesAppName(requested, it) }
        }
        return when (matches.size) {
            0 -> AppMatch(unavailable = true)
            1 -> AppMatch(app = matches.single())
            else -> AppMatch(choices = matches.sortedBy { it.label.lowercase(Locale.ROOT) })
        }
    }

    private fun normaliseAppName(value: String): String = value.trim()
        .replace(Regex("^(?:open|launch|start|play)\\s+(?:up\\s+)?", RegexOption.IGNORE_CASE), "")
        .replace(Regex("^(?:the|my|a|an)\\s+", RegexOption.IGNORE_CASE), "")
        .replace(Regex("\\s+(?:app|application)$", RegexOption.IGNORE_CASE), "")
        .trim().lowercase(Locale.ROOT)

    private fun compactAppName(value: String): String = value.lowercase(Locale.ROOT)
        .replace(Regex("[^\\p{L}\\p{N}]"), "")

    private fun matchesAppName(requested: String, app: App): Boolean {
        val target = compactAppName(requested)
        if (target.length < 3) return false
        if (compactAppName(app.label) == target) return true
        val words = app.label.split(Regex("[^\\p{L}\\p{N}]+"))
        return words.any { compactAppName(it) == target }
    }

    data class EmailRequest(val to: String, val subject: String, val body: String)
    data class EmailComposer(
        val packageName: String,
        val recipients: Set<String>,
        val ccRecipients: Set<String> = emptySet(),
        val bccRecipients: Set<String> = emptySet(),
        val subject: String?,
        val body: String?,
        val sendControlCount: Int,
    )

    fun validEmailRequest(request: EmailRequest): Boolean = email(request.to) != null &&
        request.subject.length <= 200 && request.subject.none { it == '\r' || it == '\n' } &&
        request.body.length <= 16_000

    /** All content must match before a Gmail send control can be pressed. */
    fun matchesReviewedEmail(request: EmailRequest, composer: EmailComposer): Boolean =
        validEmailRequest(request) && composer.packageName == GMAIL_PACKAGE &&
            composer.recipients.map { it.lowercase(Locale.ROOT) }.toSet() ==
                setOf(request.to.lowercase(Locale.ROOT)) &&
            composer.ccRecipients.isEmpty() && composer.bccRecipients.isEmpty() &&
            composer.subject == request.subject &&
            composer.body?.replace("\r\n", "\n") == request.body.replace("\r\n", "\n") &&
            composer.sendControlCount == 1

    const val GMAIL_PACKAGE = "com.google.android.gm"
}
