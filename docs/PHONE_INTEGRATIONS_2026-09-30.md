# Phone integrations — 30 September 2026

The Command Deck redesign was set aside at the user's request. This work adds
capabilities to the existing app and prepares a cloud trial.

## Customer-owned hosted AI

Options → Connections → Text AI provider supports the following services.
Choosing a service does not create an account, grant free credit or copy a
desktop key. Enter that service's own key on the phone. Android Keystore seals
each credential separately; changing the selected provider clears any unsaved
key field so a key typed for one service cannot follow a different selection.

| Service | Initial chat model |
| --- | --- |
| Gemini | Existing selected Gemini text model |
| Groq | `openai/gpt-oss-20b` |
| OpenRouter | `openai/gpt-oss-20b:free` |
| Cloudflare Workers AI | `@cf/google/gemma-4-26b-a4b-it` |
| Cerebras | `gpt-oss-120b` |
| Mistral | `mistral-small-latest` |
| Hugging Face | `openai/gpt-oss-120b:cheapest` |
| NVIDIA NIM | `meta/llama-3.3-70b-instruct` |
| SambaNova | `Meta-Llama-3.3-70B-Instruct` |
| Together AI | `Qwen/Qwen3.5-9B` |
| DeepSeek | `deepseek-flash` |

Model access, function calling, free allowances and charges depend on the
account and model. A model ID accepted by the input field is not proof of
account access. The app shows authentication, quota and unsupported-model
errors separately. These profiles use fixed HTTPS origins and refuse
redirects. They do not silently send an answer to another provider.

Replies and reviewed phone actions use the selected text provider. Background
microphone transcription and cloud voice still use the Gemini credential.
Android's phone voice remains available without a speech API key. The phone
does not need the PC for these configured services.

DeepSeek's phone requests explicitly select non-thinking mode: its default
thinking mode otherwise needs reasoning replay during tool turns and adds
delay. Provider reasoning fields are excluded from the displayed/spoken reply.
A token-truncated tool call is refused; truncated text remains available for
the existing continuation path. Exact tool-call identifiers survive the
action/receipt round trip, including Mistral's identifier format.

Hosted text streams no longer share the speech decoder's 4,096-event limit.
Byte limits still apply, but 4,096 token-sized text fragments plus control
events can complete and enter the existing continuation path. `[DONE]` ends
the read immediately rather than waiting for the provider to close a retained
connection. Regression tests exercise both behaviours.

Changing a text provider, Gemini key or text model cancels the Activity and
background service turns. A configuration revision blocks stale transcripts,
late replies and old tool requests; capture remains owned by the existing
foreground service and resumes with the saved listening preference. The key
removal dialogue now states that every non-Gemini text key is removed.

## Offline navigation and phone MCP

The native map adds BRouter coordinate car routes, explicit planning between
two supplied coordinates, GPX import and phone-private saved places. This
owner's phone has BRouter 1.7.9 and seven UK routing cells installed, with each
phone copy checked against its PC SHA-256 digest. See
[setup and navigation limits](OFFLINE_CAR_ROUTES_2026-09-30.md).

The phone MCP transport supports the current stateless Streamable HTTP
protocol and negotiated 2025 revisions. Discovery is bounded, reply IDs are
checked, changed tools invalidate the catalogue, and actions are never retried
automatically. Account authorisation, OAuth and unsupported multi-step
protocol workflows remain separate; see
[MCP compatibility](PHONE_MCP_INTEROPERABILITY_2026-09-30.md).

## Prepared cloud trial

The Render trial Blueprint and headless launcher are ready for later review.
The user explicitly requested **preparation only**. No cloud service, source
push or key transfer was performed. Free hosting's sleep and disposable state
are recorded in [the deployment guide](../deploy/README_RENDER_FREE.md).

For a modest paid option, Together lists Qwen3.5 9B at **US$0.17 per million
input tokens and US$0.25 per million output tokens**, with function calling.
This is a price comparison, not a measured quality or phone latency claim.
Cerebras lists GPT-OSS 120B at US$0.35 input and US$0.75 output. Free-tier and
account limits must be checked in the provider console before customer use.

Sources checked on 30 September 2026:
[Together models](https://docs.together.ai/docs/serverless/models),
[Together API compatibility](https://docs.together.ai/docs/inference/openai-compatibility),
[Cerebras public models](https://inference-docs.cerebras.ai/api-reference/models/public-models),
[Mistral chat API](https://docs.mistral.ai/api),
[Hugging Face chat](https://huggingface.co/docs/inference-providers/en/tasks/chat-completion),
[SambaNova API](https://docs.sambanova.ai/docs/en/api-reference/overview),
[NVIDIA model reference](https://docs.api.nvidia.com/nim/reference/meta-llama-3_3-70b-instruct),
[DeepSeek models](https://api-docs.deepseek.com/quick_start/pricing/).

## Device and account acceptance

Unit tests exercise request conversion, safe endpoint configuration, receipt
identifiers and incomplete-action refusal. No new provider credentials were
created or live account requests sent during those tests. Actual model access
and latency require the owner's credentials and a real phone test.

The final **3.14 / versionCode 36** compatible release was installed wirelessly
and opened through its normal launcher, retaining existing app data. APK
SHA-256: `C7B4F7A2F58BD495B609E8D3FB7149D9CCEB658CFE0502B6EE92B26444160E0B`.
A separate production-signed APK is retained; it cannot replace the existing
debug-signed installation without a deliberate credential/data migration.

The combined Android gate passed **125 tests**, with zero failures, errors or
skips. The release build and lint succeeded with **zero errors and 137
warnings**. Related backend/provider checks passed **205 tests**; source-export
checks passed **35 tests**. These gates do not verify new provider accounts.

The map displayed both UK packs and a measured GPS position. It subsequently
refused route calculation when that indoor fix exceeded 30 seconds, rather
than treating a stale position as live. Explicit coordinate planning was then
built and installed, but real BRouter route calculation, route restoration,
driving and voice timings remain **unverified on this phone**. The user deferred
all further phone fixes and device checks until returning home at that point.

The later explicit request resumes this work. The [3.15 voice/route record](PHONE_VOICE_ROUTE_FIXES_3_15_2026-09-30.md)
documents the completed source fixes, compatible APK, 139 passing Android tests
and still-outstanding wireless device/driving acceptance.

The subsequent [3.16 navigation/voice phase](PHONE_NAVIGATION_VOICE_3_16_2026-09-30.md)
adds native supplied spoken turns, locked-screen GPS and conservative BRouter
rerouting, alongside interruption/voice-session fixes. Its 174 Android and
180 related Python checks passed. Wireless installation and actual driving
remain unverified; this supersedes the historical native-guidance gaps above.
