# Teaching at your pace

ORION now defaults to one small concept per teaching reply. Each short lesson
has one goal, a small example and one check. A wrong answer gets a smaller
explanation; a correct answer gets specific feedback. **Only “next” advances.**

Start with **“Teach me in small steps”**, **“Teach me Python basics”** or a
cybersecurity/electronics topic.

| Say | What happens |
| --- | --- |
| Simpler | Another explanation of the same point |
| Repeat this step | The current step is shown again |
| Pause my lesson | Your position is saved |
| Resume my lesson | A brief recap and the same step |
| Where was I in my lesson? | Your saved topic and position |
| No time target | Pace suggestions are removed |
| Show me the full explanation | The complete reference lesson |

The suggested starting pace is five minutes per step. It is optional and has
no timer. A pause point follows three new steps by default; say “resume my
lesson” to continue. You can choose 0–10 suggested minutes and 1–5 steps per
session through `coach_settings`. There are no streaks, deadlines or penalties.

With local/offline voice, say **“Orion next”**, **“Orion simpler”** or
**“Orion A”** so the existing wake-word guard accepts a short control.
**“Orion pause my lesson”** and **“Orion resume my lesson”** save and restore
the lesson position, including while ORION is speaking.

Fifteen reviewed plans cover Python basics, secure code, recognition of cyber
threats, electronics fundamentals and hardware/firmware protection. Topic,
position and preferences are saved in ORION's private configuration. Raw answers
and medical labels are not stored by the coach. Normal conversation storage
still follows ORION's existing settings. Use separate learner ids when needed;
the coach does not infer who is speaking. A failed save is reported clearly.

For other subjects, the shared model instructions request the same short,
learner-controlled presentation. Durable step tracking currently covers the
reviewed plans. It records visited steps rather than a qualification or mastery
score. Existing module-completion tracking stays separate.

These design choices follow W3C cognitive-accessibility guidance on
[short text blocks](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o3p05-succinct-text/),
[separate instructions](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o3p09-separated-instructions/)
and [control of interruptions](https://www.w3.org/WAI/WCAG2/supplemental/patterns/o5p01-minimal-interruptions/).
They are presentation choices, not a diagnosis or a claim to treat ADHD.

The updated source is required. An existing packaged executable needs a rebuild
to include these changes.
