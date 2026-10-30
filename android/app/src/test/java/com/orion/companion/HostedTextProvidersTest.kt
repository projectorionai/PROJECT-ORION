package com.orion.companion

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class HostedTextProvidersTest {
    @Test fun everySelectableHostedProviderHasSafeConfiguration() {
        val account = "0123456789abcdef0123456789abcdef"
        assertEquals(HostedTextProviders.choices.size, HostedTextProviders.choices.distinct().size)
        for (provider in HostedTextProviders.choices.filter { it != HostedTextProviders.GEMINI }) {
            val endpoint = java.net.URI(HostedTextProviders.endpoint(provider, account)!!)
            assertEquals("https", endpoint.scheme)
            assertNull(endpoint.userInfo)
            assertNull(endpoint.query)
            assertNull(endpoint.fragment)
            assertTrue(HostedTextProviders.validModel(HostedTextProviders.defaultModel(provider)))
            assertFalse(HostedTextProviders.label(provider) == "Unknown provider")
        }
    }

    @Test fun providerEndpointsAreFixedHttpsHosts() {
        assertEquals("https://api.groq.com/openai/v1/chat/completions",
            HostedTextProviders.endpoint(HostedTextProviders.GROQ))
        assertEquals("https://openrouter.ai/api/v1/chat/completions",
            HostedTextProviders.endpoint(HostedTextProviders.OPENROUTER))
        assertNull(HostedTextProviders.endpoint("https://example.org/collect"))
    }

    @Test fun cloudflareAccountCannotAlterItsHostOrPath() {
        val account = "0123456789abcdef0123456789abcdef"
        assertEquals("https://api.cloudflare.com/client/v4/accounts/$account/ai/v1/chat/completions",
            HostedTextProviders.endpoint(HostedTextProviders.CLOUDFLARE, account))
        assertNull(HostedTextProviders.endpoint(HostedTextProviders.CLOUDFLARE, "../../other"))
        assertNull(HostedTextProviders.endpoint(HostedTextProviders.CLOUDFLARE, "account?key=leak"))
    }

    @Test fun modelIdCannotInjectAnotherUrlOrHeader() {
        assertTrue(HostedTextProviders.validModel("@cf/openai/gpt-oss-20b"))
        assertTrue(HostedTextProviders.validModel("openai/gpt-oss-20b:free"))
        assertFalse(HostedTextProviders.validModel("model\r\nAuthorization: other"))
        assertFalse(HostedTextProviders.validModel("https://example.org/collect"))
    }

    @Test fun providerKeyCannotInjectHeaders() {
        assertTrue(HostedTextProviders.validKey("example-key_123"))
        assertFalse(HostedTextProviders.validKey("key\r\nAuthorization: other"))
        assertFalse(HostedTextProviders.validKey("key with spaces"))
    }
}
