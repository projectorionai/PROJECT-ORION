# Cybersecurity and electronics learning in ORION

ORION's cybersecurity curriculum now pairs its recognition lessons and guided
labs with small, explained Python examples and exercises. Start in the normal
conversation with **“Start my cyber learning path”**. Teaching now defaults to
one small step at a time, beginning with Python basics for a new learner. No extra package, account
or external target is needed for the bundled examples.

## What to ask

| Request | Curriculum tool call |
| --- | --- |
| Start my cyber learning path | `action="path"` |
| List the security lessons | `action="lessons"` |
| Teach me SQL injection and its fix | `action="teach", topic="sql_injection"` |
| Show me secure SQL code | `action="code", topic="sql_injection"` |
| Give me an access control exercise | `action="exercise", topic="access_control"` |
| My answer to this step is B | `action="coach_answer", answer="B"` |
| Teach me electronics fundamentals | `action="teach", topic="electronics_fundamentals"` |
| Explain protection against destructive hardware | `action="teach", topic="hardware_protection"` |
| Explain signed firmware updates | `action="teach", topic="firmware_security"` |
| Give me a simpler explanation | `action="coach_simpler"` |
| Next step | `action="coach_next"` |
| Pause my lesson / resume my lesson | `action="coach_pause"` / `action="coach_resume"` |
| No time target for my lessons | `action="coach_settings", pace_minutes=0` |
| Show the complete secure SQL example | `action="code", topic="sql_injection", format="full"` |

Short lessons save their topic, position and pacing preferences, so you can
resume after restarting ORION. Answering, repeating and simplifying do not
advance the lesson. Say **“next”** when you want another step. By default, a
pause point follows three new steps; there is no timer. Detailed reference
lessons remain available with `format="full"`. Their original A/B/C exercise
uses `action="check", topic=..., answer=...`, separately from the current-step
`coach_answer`. See the [short-step teaching guide](SHORT_STEP_TEACHING.md).

The examples cover parameterised SQLite queries, escaping text for an HTML
context, server-side ownership checks, and analysis of synthetic login events.
Each explains its code, gives expected output, and offers a fixed-choice
exercise with feedback. The electronics lesson explains voltage, current,
resistance and power using a low-voltage LED simulation.

The tool displays code as text. It does not execute the examples, pasted code
or answers. The bundled examples use the Python standard library and synthetic
data; their execution and expected results are checked in the automated tests.
These short examples demonstrate particular controls, not complete production
security. HTML text escaping does not cover JavaScript or URL contexts, and
login-event thresholds are investigation clues rather than proof of an attack.

## From a lesson to a lab

Use the existing `pentest_lab` tool for guided SQL injection, XSS, port scanning
or privilege-escalation practice on training targets. A named range is an
educational pointer. A training provider's main website is not an authorised
target. Actual hosts must be loopback or explicitly recorded in ORION's
authorised-target list. Private IP addresses and `.local` names require the
same authorisation as other hosts. ORION's recorded scope is a technical
allowlist; it does not establish permission from the system owner.

Record the agreed target, methods, evidence, impact, fix and retest outcome for
each practice task. This learning approach follows the emphasis on scoping,
reporting and follow-up in the
[NCSC penetration-testing guidance](https://www.ncsc.gov.uk/guidance/penetration-testing).

## Hardware defence and professional development

Hardware teaching covers ESD, abnormal input power, untrusted USB devices,
signed firmware updates, recovery and debug-access controls. It includes no
construction instructions, circuits or component specifications for shock
devices or devices designed to destroy electronics. Requests to build those
devices receive a brief boundary and a route back to fundamentals or defence.
This boundary also appears in the full and compact model instructions; prompt
instructions alone are not a security boundary for arbitrary generated replies.

The existing module-completion tracking remains available. Short lessons record
which steps were visited, without claiming that they were mastered. The exercises
give feedback without recording a qualification or marking a module complete.
Learning content does not retrain the language model or confer accreditation,
employment eligibility or security clearance. For UK training pathways, consult
the current [NCSC cyber-skills resources](https://www.ncsc.gov.uk/cyber-skills).

The technical lessons link directly to Python, OWASP, NIST and manufacturer
documentation. Use current employer requirements and official recruitment
information when planning a defence or intelligence career.
