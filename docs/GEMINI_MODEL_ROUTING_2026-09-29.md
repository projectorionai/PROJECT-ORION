# Gemini model routing — 29 September 2026

ORION uses a Gemini key through several distinct API paths. A model in AI
Studio's quota table is not automatically a chat model, and a `0 / 0` row does
not grant usable requests. Quotas can change independently of this file;
ORION responds to HTTP 429 by cooling or rotating the affected model rather
than hardcoding the day's usage figures.

| Work | ORION path | Default | Alternates or constraint |
| --- | --- | --- | --- |
| Everyday text and function calls | `gemini_text`, OpenAI-compatible chat | `gemini-3.5-flash-lite` | 2.5 Flash-Lite, 2.5 Flash, 3.1 Flash-Lite |
| Demanding coding, reasoning and writing | `gemini_reasoning`, OpenAI-compatible chat | `gemini-3.8-flash` | 3.7, 3.6, 3.5, 3 Flash Preview, 2.5 Flash |
| Hosted Gemma 4 text and image input | `gemma_4_31b` and `gemma_4_26b`, native `generateContent` | 31B for complex work; 26B for routine fallback | Both share the configured Gemini key but have separate model quotas |
| Still-image understanding | Existing vision router | The active vision-capable text model, with the `electronics.vision_model` override | OCR remains the last fallback |
| Video frames | Existing vision router, then direct SDK fallback | `gemini-3.5-flash-lite` | `ORION_VIDEO_MODEL` can override the direct fallback |
| Google Search grounding | Search backend | `gemini-3.5-flash-lite` | 2.5 Flash-Lite, then 3.1 Flash-Lite; `ORION_SEARCH_MODEL` overrides the first choice |
| PC live conversation | Separate Gemini Live WebSocket worker | `gemini-2.5-flash-native-audio-preview-12-2025` | See `orion_core/constants.py` for Live fallbacks |
| Phone Charon speech | Dedicated Live WebSocket, then speech generation | `gemini-3.8-live` | Direct 3.8 Flash TTS and other compatible TTS models if Live fails before playback; Options can select direct speech |
| Explicit semantic text comparison | Native `embedContent` requests | `gemini-embedding-2` | Whole-request fallback to `gemini-embedding-001`, then local comparison |
| Delegated public web research | Separate Interactions API | `antigravity-preview-09-2026` | Uses 3.5 Flash-Lite under a bounded budget; requires an explicit public topic |

The 3.1 Flash-Lite model is listed last for routine text. It supports text and
function calling according to Google, but an ORION-shaped streamed request
returned HTTP 503 during this audit. The provider will fall through if it is
unavailable. User-saved model lists and primary model choices are retained;
only the exact former shipped defaults migrate to this routing. A custom text
profile also prevents the new reasoning profile from silently taking over its
complex requests.

The standalone phone starts everyday turns with 3.5 Flash-Lite and tries 3.8,
3.7 and 3.6 Flash for demanding questions. A user-selected compatible text
model remains first. A model refused before any reply or action can fall
through; subsequent function turns remain on the model that began them. Daily
429 cooldowns persist on the phone until the next Pacific midnight.

The key alone cannot make every listed model usable. Live and Transcribe Live
models use the Live protocol. TTS models need speech generation, embedding
models need an embeddings endpoint, image models need an image-output handler,
and hosted agents use an agent API. ORION has a consent-gated semantic text
comparison using Gemini Embedding 2 or Embedding 1 and a separate public-topic
Antigravity research tool. It does not yet have a Gemini image-generation
handler. These task-specific models are excluded from text fallbacks, as are
the screenshot's zero-quota models. Further model support requires a matching
task, API adapter, account quota, and result handling.

Gemma 4 26B and 31B are hosted by the Gemini API. ORION routes them through
`orion_core/gemma_native.py`, which builds native `generateContent` requests,
parses streamed text and function proposals, and never executes a proposed
function. The existing desktop tool approval and dispatcher boundary remains
responsible for action execution and its verified receipt. A separate
`ProviderRouter.propose_gemma_functions` method exposes proposals to an
authorised caller; ordinary conversation sends no function declarations and
cannot turn a proposal into a claim that an action ran. Existing custom text
model choices keep the new Gemma profiles disabled until the owner enables
them explicitly. Both Gemma profiles use minimal thinking for routine replies
so the hidden thinking tokens do not consume a short output budget. Setting
`ORION_MODEL_THINKING=1` enables high thinking, and an individual native request
can set `thinking_level` to `high`. An empty reply that ends at `MAX_TOKENS` is
treated as a failed attempt so the provider router can use its normal fallback
path. A single bounded live 26B request with a 128-token output budget returned
`ORION ready.` with `STOP`; this validates that request shape, not ongoing
service availability. Small 31B requests with `thinkingConfig=minimal`
returned `STOP` through direct REST, including the complete adapter-shaped
payload. Other 31B requests returned HTTP 500 or timed out, including both
JSON and SSE calls through the native adapter. No single incompatible field
was reproducible; the router's existing failure cooldown and fallback remain
necessary while that hosted model is intermittent.

References: [Gemini model catalogue](https://ai.google.dev/gemini-api/docs/models),
[Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite),
[Gemini 3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash/),
[video understanding](https://ai.google.dev/gemini-api/docs/video-understanding),
[Gemma on the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api),
[Gemini deprecations](https://ai.google.dev/gemini-api/docs/deprecations), and
[Live API](https://ai.google.dev/gemini-api/docs/live).
