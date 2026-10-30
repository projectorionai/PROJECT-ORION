# Antigravity public research

ORION exposes `antigravity_research` for an explicit request to delegate a
**public web research** topic. The handler uses Google's separate Interactions
API with `antigravity-preview-09-2026`. This agent is not a chat model and is
not added to the ordinary text model fallback list.

Each call sends one 8–240 character topic to a fresh remote sandbox. Google
Search and URL Context are the only explicitly supplied tools; Google also
provides the sandbox's own scratch filesystem. ORION does not mount local
files, memory, desktop actions, phone data, account integrations or MCP servers. The
topic validator rejects obvious paths, local addresses and credentials. The
tool declaration tells the model to call this only for explicit delegated
public research; other questions continue through ORION's normal providers.

The request selects `gemini-3.5-flash-lite` as the underlying model, because
the user's AI Studio snapshot shows unused quota for it. The agent has a
20,000 total-token budget (input, output and thinking), a two-minute local
deadline, five-second asynchronous polling and an attempted remote cancellation
on timeout or local cancellation. Google's token budget is best-effort and may
slightly overshoot. An incomplete result is labelled partial; ORION never
continues the agent automatically with a new budget. Preview usage may incur
charges under the Google project. Errors do not echo API response bodies or
the credential. A failed or exhausted request is reported plainly.

The implementation is tested with fake interactions; it has **not** spent the
user's agent quota on a live call. Google may change the preview API schema.
Deep Research Pro Preview has 0/0 allowance in the supplied snapshot and is
not callable with this key merely because it appears in the catalogue.

References:

- [Google Antigravity agent](https://ai.google.dev/gemini-api/docs/antigravity-agent)
- [Managed agents quickstart](https://ai.google.dev/gemini-api/docs/managed-agents-quickstart)
- [Background execution](https://ai.google.dev/gemini-api/docs/background-execution)
