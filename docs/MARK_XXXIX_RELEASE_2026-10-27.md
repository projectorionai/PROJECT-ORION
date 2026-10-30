# ORION Mark XXXIX — Sentinel

Package identity: 39.0.0. Build identity: 2026.10.27-r2-xxxix.

## Manner

- ORION's manner is modelled on JARVIS from Iron Man: a refined British AI
  butler, unflappable and precise, with a bone-dry wit. Roughly one relaxed
  reply in three carries one short wry touch — understatement, deadpan, a
  polite jab when you are being lazy or up far too late — and then he does the
  job. No wit when you are upset, unwell or stressed, during errors, safety,
  health or money, or when you ask him to be serious.
- Gushing is gone: no "Ooh", "Yay", "Okie-dokie", "brilliant question!" or
  "sweet dreams!". Feelings are shown in a few composed words.
- Common sense: he reads the hour (after midnight it is late, and he says so),
  answers the question in his first sentence, and points out an obvious snag
  once before helping.
- His own fixed lines and goodbyes gained a few dry ones ("Get some sleep,
  sir. I'll mind the fort.").

## Voice and singing

- One manner in one voice: professional. The cute manner is retired; asked for
  it, he declines and stays as he is. A stored "cute" setting now reads as
  professional.
- Delivery: crisp, measured, a refined British accent, understated — never
  bubbly or sing-song.
- Singing sounds sung, not tuned: notes are approached from just below, leaps
  overshoot a hair and settle, the pitch wanders a little as a voice does,
  each long note has its own vibrato that blooms after it settles, a phrase's
  last note falls away, onsets breathe around the beat, lines rise and ease
  off in loudness, held vowels no longer buzz, there is an in-breath before a
  new line, and words within a line join up (legato). Measured on his offline
  voice singing Happy Birthday: silence inside lines 4.0% → 2.0%, pitch
  movement in long notes 15.1 → 18.6 cents, tuning still within 6 cents.

## Coming back

- Every clean exit leaves a note. The next start knows whether it is a restart
  ("Back in 9 seconds"), a quick return after a shut-down (a short "back so
  soon" — no weather, no briefing offer), or the first start on a new Mark (the
  usual greeting, plus "I'm running Mark XXXIX now").
- He keeps the words he actually opened with and is shown his last few, so no
  opening, phrase or detail repeats start after start.

## Face

- Five Screen Face designs. **Sentinel** (the new default) is drawn for the
  crimson-and-silver HUD: wide, low eyes with a silver edge and a lit core, a
  light-bar mouth that becomes a level meter when he speaks, no blush marks or
  sparkles. **Arc** has ring eyes with a lit pupil; **Visor** has slit eyes and
  a waveform mouth; **Minimal** has small round eyes; **Companion** is the
  previous look.
- Every design wears every expression. Choose in Preferences (Face Design) or
  say "switch your face to arc", "use your visor face", "go back to your old
  face".

## Plugins

- The five plugins reported every boot as "rejected" (json_validator,
  thought_model_monitor, dependency_manager, viseme_argument_adapter,
  viseme_debugger) are retired: archived to `custom_tools/_superseded`, never
  deleted. The boot line now separates failures from plugins that are switched
  off, retired or quarantined.
- Rebuilt into ORION:
  - **Telegram remote** (was jv_telegram_remote): write to ORION from Telegram.
    Off until configured — a bot token and an allow-list of chats in
    `config/messaging.json` (`"telegram": {"remote": {"enabled": true, "chats":
    [your chat id]}}`). Private chats only; anything else, and anything sent
    while ORION was off, is ignored. Each message goes through the phone
    gateway's own checks; approvals are never taken over Telegram.
  - **Mood over time** (was emotional_insight_generator): "how have I been
    this week?" — counted from your own words in memory, with one suggestion.
  - **Settings and JSON check** (was json_validator): "are my settings files
    sound?" checks every settings file; JSON (and a JSON Schema) can be checked
    too. No extra package needed.

## Fixes

- First-chance COM exceptions logged by Windows during a clean shut-down
  (0x8001010e) were filed as crashes, and each sent a Reliability Agent to
  investigate. They are no longer reported.
- The restart note is read off the event loop (it blocked it for 122 ms).
