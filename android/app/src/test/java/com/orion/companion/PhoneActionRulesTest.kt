package com.orion.companion

import org.junit.Assert.*
import org.junit.Test

class PhoneActionRulesTest {
    @Test fun webLinksRejectCredentialsAndNonWebSchemes() {
        assertEquals("https://example.org/path?q=one", PhoneActionRules.webUrl("https://example.org/path?q=one"))
        listOf("javascript:alert(1)", "file:///secret", "https://user:pass@example.org", "https://example.org\n").forEach {
            assertNull(PhoneActionRules.webUrl(it))
        }
    }

    @Test fun appLabelsRequireOneExactInstalledMatch() {
        val apps = listOf("Mail" to "first.mail", "Mail" to "second.mail", "Gmail" to "com.google.android.gm")
        assertNull(PhoneActionRules.exactPackage("Mail", apps))
        assertNull(PhoneActionRules.exactPackage("Gm", apps))
        assertEquals("com.google.android.gm", PhoneActionRules.exactPackage("gmail", apps))
    }

    @Test fun naturalAppNamesResolveOnlyUniqueInstalledLaunchers() {
        val apps = listOf(
            PhoneActionRules.App("YouTube", "com.google.android.youtube"),
            PhoneActionRules.App("YouTube Music", "com.google.android.apps.youtube.music"),
            PhoneActionRules.App("Google Chrome", "com.android.chrome"),
        )
        assertEquals("com.google.android.youtube",
            PhoneActionRules.resolveApp("open the You Tube app", "", apps).app?.packageName)
        assertEquals("com.android.chrome",
            PhoneActionRules.resolveApp("Chrome", "", apps).app?.packageName)
        assertNull(PhoneActionRules.resolveApp("You", "", apps).app)
        assertTrue(PhoneActionRules.resolveApp("You", "", apps).unavailable)
    }

    @Test fun gameCategoryOpensUniqueGameAndDisambiguatesSeveral() {
        val puzzle = PhoneActionRules.App("Word Puzzle", "uk.example.puzzle", isGame = true)
        val racing = PhoneActionRules.App("Racing", "uk.example.racing", isGame = true)
        val browser = PhoneActionRules.App("Browser", "uk.example.browser")
        assertEquals(puzzle, PhoneActionRules.resolveApp("open a game", "", listOf(puzzle, browser)).app)
        val choice = PhoneActionRules.resolveApp("one of my games", "", listOf(puzzle, racing, browser))
        assertNull(choice.app)
        assertEquals(setOf("Racing", "Word Puzzle"), choice.choices.map { it.label }.toSet())
        assertTrue(PhoneActionRules.resolveApp("game", "", listOf(browser)).unavailable)
    }

    @Test fun ambiguousNamesAndConflictingPackagesNeverSelectAnApp() {
        val apps = listOf(
            PhoneActionRules.App("Notes", "uk.example.notes"),
            PhoneActionRules.App("Notes", "uk.example.notes.other"),
            PhoneActionRules.App("YouTube", "com.google.android.youtube"),
        )
        assertEquals(2, PhoneActionRules.resolveApp("Notes", "", apps).choices.size)
        assertTrue(PhoneActionRules.resolveApp("YouTube", "uk.example.notes", apps).unavailable)
        assertTrue(PhoneActionRules.resolveApp("YouTube", "uk.example.youtube.music",
            apps + PhoneActionRules.App("YouTube Music", "uk.example.youtube.music")).unavailable)
        assertTrue(PhoneActionRules.resolveApp("", "uk.example.missing", apps).unavailable)
        assertTrue(PhoneActionRules.resolveApp("", "bad.package;intent", apps).unavailable)
        assertEquals("com.google.android.youtube",
            PhoneActionRules.resolveApp("YouTube", "com.google.android.youtube", apps).app?.packageName)
    }

    @Test fun handsFreeLaunchPolicyOnlyCoversResolvedAppsAndOsmAnd() {
        val app = PhoneActionRules.resolveApp("Calculator", "",
            listOf(PhoneActionRules.App("Calculator", "com.android.calculator2"))).app
        assertNotNull(app)
        assertTrue(PhoneActionRules.mayLaunchWithoutReview("open_app", app!!.packageName))
        assertTrue(PhoneActionRules.mayLaunchWithoutReview("navigate", "net.osmand"))
        assertTrue(PhoneActionRules.mayLaunchWithoutReview("map", "net.osmand.plus"))
        assertFalse(PhoneActionRules.mayLaunchWithoutReview("navigate", "com.google.android.apps.maps"))
        assertFalse(PhoneActionRules.mayLaunchWithoutReview("open_app", "not a package"))
        for (kind in listOf("make_call", "send_sms", "send_email", "capture_dom", "click",
            "set_text", "mcp_authorise", "share", "open_url", "home")) {
            assertFalse(kind, PhoneActionRules.mayLaunchWithoutReview(kind, "com.android.calculator2"))
        }
    }

    @Test fun reviewedEmailCannotChangeRecipientContentOrClient() {
        val request = PhoneActionRules.EmailRequest("person@example.org", "Reviewed subject", "Exact body\nSecond line")
        val composer = PhoneActionRules.EmailComposer(PhoneActionRules.GMAIL_PACKAGE,
            setOf(request.to), subject = request.subject, body = request.body, sendControlCount = 1)
        assertTrue(PhoneActionRules.matchesReviewedEmail(request, composer))
        listOf(
            composer.copy(recipients = setOf("another@example.org")),
            composer.copy(recipients = setOf(request.to, "another@example.org")),
            composer.copy(ccRecipients = setOf("another@example.org")),
            composer.copy(bccRecipients = setOf("another@example.org")),
            composer.copy(subject = request.subject + " changed"),
            composer.copy(body = request.body + " changed"),
            composer.copy(packageName = "another.mail"),
            composer.copy(sendControlCount = 2),
            composer.copy(body = null),
        ).forEach { assertFalse(PhoneActionRules.matchesReviewedEmail(request, it)) }
    }

    @Test fun recipientMustBeOneAddressAndPhoneNumberMustBeExact() {
        assertNull(PhoneActionRules.email("one@example.org,two@example.org"))
        assertNull(PhoneActionRules.email("one@example.org\r\nBcc:two@example.org"))
        assertEquals("+447700900123", PhoneActionRules.number("+44 7700 900-123"))
        assertNull(PhoneActionRules.number("Sam"))
        assertFalse(PhoneActionRules.validEmailRequest(PhoneActionRules.EmailRequest("one@example.org", "subject\nBcc", "body")))
    }
}
