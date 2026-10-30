# Mark XXXIII — speech continuity audit

## Faults found and corrected

1. **Phone speech fetched sentences after their predecessors had finished playing.**
   Both direct Charon and PC-connected speech created an `AudioTrack` for each
   section, drained it, released it and only then requested the next section.
   Each join therefore included another provider round trip and device setup.
   There is now one output device per reply and a separate ordered PCM worker.
   Synthesis can request the following section whilst the preceding audio plays.
   Waiting sentences are joined within the provider's existing 1,200-character
   limit, whilst the opening sentence can still start promptly.

2. **The PC fallback repeated the same wait between long sections.**
   ElevenLabs sections now feed the same persistent renderer in order without
   draining it between sections. The local synthesis worker keeps a single
   speaking window over the queued reply and drains at its end. A switch to a
   local fallback waits for earlier PCM before speaking. Producer back-pressure
   keeps fetching ahead without retaining an unlimited answer in memory.

3. **A temporary gap in Gemini Live audio reopened the PC microphone.**
   Device-tail completion was treated as completion of the entire answer. The
   state machine now tracks an unfinished Live producer separately from device
   playback. A short gap keeps the same speaking window and microphone gate.
   `turn_complete` releases the producer; the final queued device audio remains
   reserved until it drains, even if completion arrives before playback starts.
   The existing watchdog releases a lost producer after 28 seconds with no PCM,
   and session teardown flushes the final partial frame. Its progress check uses
   actual playback, so the producer reservation cannot indefinitely hide a stall.

4. **Holding cloud speech could replay a whole section over preserved PCM.**
   Shared ElevenLabs playback already retained an exact PCM position, but the
   text synthesiser also retained the full sentence for resynthesis. A shared
   hold now finishes fetching the current bounded section and resumes its
   preserved PCM. The standalone fallback retains its existing replay behaviour.

5. **Late provider audio could cross an explicit interruption.**
   A provider callback between renderer clearing and the text worker observing
   its stop could previously tag old PCM with the new interruption generation.
   Cloud streams now retain their starting generation. The renderer rejects
   mismatched generations, and generation change plus queue clearing is atomic.

## Verification

- Focused Python regressions: **101 passed**. They cover PCM ordering, short and long silence,
  native producer gaps, missing completion, final frames, cloud section
  prefetch, interruption generations, hold/resume and shutdown ordering.
- Android `testDebugUnitTest`: **67 tests, zero failures, zero errors, zero skips**.
  New playback tests hold the output callback whilst later sections arrive,
  verify final samples over packet boundaries, exercise cancellation and require
  failed output to report a fault instead of completing silently.
- Android builds use the existing installed Gradle 9.5.0 and Android Studio JBR.
  Provider credentials and provisioning remain intact; the first continuity
  revision retained Charon. The follow-up below adds explicit voice choices.
- The phone PCM queue is bounded at roughly 175 seconds, with back-pressure.
  Cancellation clears waiting samples, interrupts its worker and waits at most
  500 ms for release; device writes check cancellation between small slices.

## Remaining limits

These checks are offline and do not measure the installed phone's speakers or
the live provider. Real-device listening is still needed after installing the
integrated build. A provider that sends audio slower than real time can still
produce a gap; these changes remove application-imposed request and device gaps.
Quota refusal and a lost partial stream retain the complete reply as text. PCM
has no trustworthy word offset, so a partially spoken section is never blindly
replayed through a second voice. The user subsequently authorised phone speech
when cloud audio is unavailable; the following revision implements that policy.

The speech modules also retain pre-existing broad exception and style findings
under the repository's strict Ruff profile; that profile is not a clean gate.

## Follow-up: paragraph pauses, voice choice and authorised fallback

- Short paragraphs previously produced many small synthesis requests. The
  opening completed sentence still starts promptly; following streamed text is
  grouped into approximately 220–600-character sections. Already queued sections
  can still combine within the established 1,200-character request limit.
  Paragraph whitespace is flattened for speech while the terminal retains the
  original layout and complete answer.
- A generated pause was previously sent to the phone unchanged. Sustained
  near-silence now retains at most 260 ms across packet and section boundaries.
  This operates on 10 ms frames with peak amplitude at most 120 out of 32,768.
  All louder frames, normal shorter pauses and incomplete final frames remain.
  A synthetic 1,500 ms silent paragraph gap becomes 260 ms: **1,240 ms removed**.
  This is a conservative threshold rather than a speech detector; unusually
  quiet sustained material below it can be shortened. Network time without
  supplied audio cannot be trimmed.
- **Options → Voice → Voice** offers the documented Gemini voice catalogue and
  an explicit **Phone** choice. Selection is stored using the existing AES-GCM
  Android Keystore store. Both direct speech and warm Live setup use it; the
  paired host validates a requested cloud voice and returns its actual name in
  `X-Orion-Voice`. Changing voice cancels the old reply and warms a new session.
- Google lists the same thirty prebuilt names for the Developer API Live route
  and speech generation. The menu uses this supported set; model access, key
  permissions and quota still determine availability. A failed Live setup uses
  the existing bounded cooldown rather than waiting on each section. Sources:
  [Live voice configuration](https://firebase.google.com/docs/ai-logic/live-api/configuration?hl=en#specify-a-response-voice)
  and [Gemini speech voices](https://ai.google.dev/gemini-api/docs/generate-content/speech-generation?hl=en#voice-options).
- The selected cloud voice remains preferred. A refusal or connection failure
  **before any PCM for that section** falls back to normal phone speech. After
  partial PCM, the remaining audio drains and only subsequent untouched sections
  use the phone engine. Its speaking gate stays held until the completion
  callback; cancelled or late callbacks cannot release a newer utterance.
  No error is read aloud automatically. An empty completed Live/host stream is
  unavailable, not a successful spoken reply.
- Explicit phone speech works without a Gemini key, including typed replies
  using another configured text provider. It uses the installed Android speech
  engine and available British English voice data. Background transcription
  retains its separate provider requirements. Voice fallback, request count and
  shortened silence appear only in Performance diagnostics, without retaining
  prompts, transcripts, audio or keys.

### Follow-up verification

- Android compilation and `testDebugUnitTest`: **74 tests passed**, no failures,
  errors or skipped tests. New tests cover paragraph grouping and complete final
  words, packet-independent silence shortening, quiet audible samples, voice
  catalogue validation and cancellation/late-callback microphone gating.
- Focused Python speech tests: **27 passed**. These exercise both current and
  legacy selected-voice request schemas, authentication and input rejection,
  host streaming headers and framing, empty audio refusal, PCM continuity and
  complete long replies.
- Authorised host probes measured first PCM at **4.149 s** for the legacy
  `gemini-2.5-flash-preview-tts` request; its three chunks arrived together.
  A complete WAV request took **2.546 s**. A separate configured
  `gemini-3.8-live` probe delivered first PCM **0.807 s after the turn was sent**
  (1.115 s including setup). These are individual short host requests, not
  installed-phone or long-answer timings. Reports are in
  `reports/mark_xxxiii_2026_09_30/charon-live-provider-timing.json` and
  `reports/mark_xxxiii_2026_09_30/live-3_8-provider-timing.json`.
  An older 2.5 Live compatibility probe completed with no PCM; that failure is
  now correctly treated as unavailable. No measurement proves all voices or
  devices have identical latency. Long responses still require device testing.
