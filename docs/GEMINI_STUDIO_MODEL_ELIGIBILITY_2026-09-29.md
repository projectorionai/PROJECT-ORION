# Google AI Studio model eligibility — 29 September 2026

The owner's quota screenshot records **usage at one moment**, not a promise of
future allowance. ORION must use each model through its matching API and task.
It does not send an ordinary chat prompt to an image, audio, embedding or agent
model. Google counts limits by model and project; a `0 / 0` daily allowance
cannot be made usable by switching the request format.

| Quota-table entries | ORION capability | Current handling |
| --- | --- | --- |
| Gemini 2.5 Flash, 2.5 Flash-Lite, 3 Flash, 3.1 Flash-Lite, 3.5 Flash, 3.5 Flash-Lite, 3.6 Flash, 3.7 Flash, 3.8 Flash | Text generation and tool planning | Eligible models rotate by task and availability. The paired phone uses 3.5 Flash-Lite first and stronger Flash models for demanding turns. Quota refusals cool the specific model. |
| Gemma 4 26B and 31B | Hosted text and still-image understanding | Desktop native `generateContent` profiles share the Gemini key and can be selected for suitable work. Their function calls remain proposals for ORION's normal approval path. |
| Gemini 3.1 Flash TTS, 3.8 Flash TTS, 3.8 Flash Lite TTS, 2.5 Flash TTS | Speech generation | The phone attempts Charon on compatible TTS models if Live speech is unavailable before playback; exhausted models cool down. PC-linked speech uses 3.8 Flash TTS by default. |
| Gemini 2.5 Flash Native Audio Dialog, 3 Flash Live, 3.8 Live and 3.8 Live Extended Thinking | Live audio | PC live conversation uses 2.5 Native Audio Dialog; standalone phone Charon uses 3.8 Live. The other variants are **not** interchangeable without testing their session and audio behaviour. |
| Gemini Embedding 1 and 2 | Semantic similarity | `semantic_compare` accepts explicit short texts, with a consent gate for the cloud call. Local comparison is the fallback. |
| Antigravity | Delegated public research | `antigravity_research` uses the separate Interactions API with a bounded 3.5 Flash-Lite agent run. No local data or ORION tools are mounted in its sandbox. |
| Gemini 3.5 Transcribe, 3.5 Transcribe Live, 3.5 Live Translate | Transcription or translation | Listed with allowance, but no standalone ORION adapter is yet verified. Android speech recognition and the existing PC Live audio path remain active. |
| Deep Research Pro Preview, Gemini 2 Flash and 2 Flash-Lite, Computer Use Preview, Nano Banana image variants, Gemini 2.5 Pro and Pro TTS, Gemini 3.1 Pro, Gemini Omni variants, Gemini Robotics ER 2, Lyria variants, Veo variants | Their respective specialised tasks | The screenshot shows a zero daily allowance. They are excluded from automatic routing. They need account access, task-specific adapters and verified result handling before ORION can offer them. |

Search and Maps grounding rows are **tool allowances**, not additional chat
models. ORION's eligible search grounding runs separately from ungrounded text
generation. A displayed zero allowance is respected. The phone's offline
navigation capability does not depend on a Google Maps grounding request.

Evidence: [Google rate limits](https://ai.google.dev/gemini-api/docs/rate-limits),
[model catalogue](https://ai.google.dev/gemini-api/docs/models),
[Live API](https://ai.google.dev/gemini-api/docs/live),
[embeddings](https://ai.google.dev/gemini-api/docs/embeddings),
[Gemma on Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api),
and [Antigravity](https://ai.google.dev/gemini-api/docs/antigravity-agent).
