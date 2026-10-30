# ORION Mark XXXVIII — Clean Deck

Package identity: 38.0.0. Build identity: 2026.10.27-r1-xxxviii.

## Command Deck

- **Toolkit** is now one page with two tabs: *Tools* (margin, ads, currency,
  revenue log, ideas, protocols) and *Focus* (focus blocks and study). The old
  Cognition page and the Toolkit's own focus timer were the same thing twice.
- **Diagnostics** has two tabs: *Health* (self-checks, faults, crashes) and
  *Load* (the live graphs and processes that used to be the Telemetry page).
- "Open the toolkit", "show telemetry", "open focus" and the other old names
  still work, and land on the right tab.
- The one-line descriptions under page titles, the Vision Lab introduction,
  long example placeholders and explanatory notes in Preferences are gone.
  Titles, live status and real warnings stay.
- The side panel shows processor, memory, network, graphics and heat, and one
  line underneath: *ORION has been active for 3 h 14 min*. The status tiles
  only appear when something needs attention (recording, a security finding,
  a running routine, a mind that is not ready).

## Voice and manner

- ORION has one voice: Charon online, George offline. Professional and cute
  are two ways of talking in that same voice. Women's voices (and Puck) are
  refused, and old settings that named one are moved to his voice.
- "Change your voice back" switches to the professional manner instead of
  "I can't change my voice".
- He checks sums and facts before agreeing and says kindly when something is
  wrong (he had called f(2) = 1 correct for f(x) = 2x² − 3). The cute manner
  no longer means flattery.

## Expressions

- Being told he is bad at something ("you're bad at singing") shows anger;
  frustration in his own reply does too.
- While he works on something his face shows focus.
- "I'm going to shut you down" is no longer read as an emergency; only an
  outage ("the site is down") is.

## Agents and tasks

- A specialist that gets no answer from any model says so in one plain
  sentence. It used to read out the router's chained error and its step list.
- Specialists and consults tell the model router how much of their time is
  left, so a slow reasoning model leaves room for a quicker one. All eleven
  agents were run end to end.
- Finished background tasks are announced as "The Reliability Agent has
  finished. …", without the tool tags.
- Tasks page: shorter placeholders and status lines; the empty page just says
  *No tasks yet*.
- Overdue reminders say "3 days", not "72h", and entries more than two weeks
  past their date (trips, days that have been) are no longer nagged about.

## Fixes

- Memory search: the full-text index rejected queries with punctuation
  (`fts5: syntax error near ","`) and fell back to a slow scan. Queries are
  now plain words only.
- Shutdown: the intent learner is stopped and joined before exit, and the
  process ends with TerminateProcess, so compiled libraries are no longer
  unloaded under running threads (the native faults in the crash reports).
- The sentinel no longer reports ORION's own processes as unfamiliar.
- Plugin choices made in 37.5 are applied from the owner's own plugins file
  (they had been written to the source folder's copy).
