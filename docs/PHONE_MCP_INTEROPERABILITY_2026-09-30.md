# Phone MCP interoperability

The Android app connects directly to one user-configured HTTPS MCP endpoint. It does not need the PC to be online. Configure it under **Options → Connections → MCP** with the service's exact Streamable HTTP endpoint and, where required, a bearer token. The token remains encrypted with Android Keystore. Configure only a service and account you intend ORION to access.

## Protocol support

- `2026-07-28`: the phone attempts a read-only `tools/list` with per-request protocol, implementation and empty optional-capability metadata. Matching `MCP-Protocol-Version` and `Mcp-Method` headers are sent. Discovery needs one request when the tool list fits on one page.
- `2025-11-25`, `2025-06-18`, `2025-03-26`: a recognised version refusal advertising a supported legacy revision, or an unrecognised HTTP 400 from the discovery probe, permits an `initialize` handshake. Subsequent requests use the negotiated revision and any server-issued session identifier. The client respects the negotiated `tools` capability; a server offering only resources is connected with zero callable tools.
- JSON replies and request-scoped SSE replies are supported. Empty SSE priming events and progress notifications are accepted. Changed-tool notifications invalidate cached declarations. Result identifiers are checked without converting strings or fractional numbers into integers.
- Modern `x-mcp-header` annotations are validated before advertising tools. Values are read from their exact nested property paths, primitive types and integer ranges are checked, and Unicode, whitespace or sentinel values are encoded using the protocol's Base64 format. Invalid annotations exclude the affected tool; the connection status reports the count.

The client does not claim support for the deprecated two-endpoint HTTP+SSE transport, OAuth sign-in, resources/prompts interfaces, server-initiated sampling or elicitation, subscription streams, resumable legacy SSE streams, or modern multi-round-trip input requests. A service requiring these features needs an additional integration. Unsupported or incomplete results are not reported as completed actions.

## Connection and execution behaviour

Discovery has one six-second deadline across negotiation and pagination, a five-minute in-memory cache, a thirty-second failure pause and limits of ten pages and one hundred declarations. Cancelling a phone turn cancels its active HTTP call. JSON responses are limited to one MiB and SSE responses to two MiB. Tool arguments and result forwarding retain their existing review limits.

Every tool call still needs the existing local review. The saved configuration revision is checked before and after the request; changing the endpoint or token invalidates an approval for the previous connection. The app never automatically repeats a tool call after a timeout, HTTP error, missing result, expired session or protocol error. A lost reply can mean the service acted without a verified receipt, so a second attempt needs a fresh user decision.

Legacy sessions receive a best-effort HTTP DELETE when discarded or closed. It runs off the UI thread with its own dispatcher, connection pool and two-second deadline. A server may refuse DELETE; cleanup neither blocks a new phone turn nor causes another tool call.

No account, token or public server is invented or enabled by this change. A configured endpoint is not proof that its tools have been discovered or that an action has completed.

## Verification

`PhoneMcpTransportTest` uses in-process OkHttp interceptors: no production service, credential or external action is involved. The cases cover stateless discovery, legacy negotiation, capability handling, session propagation and cleanup requests, authentication and protocol failures, response correlation, SSE priming, changed declarations, incomplete results, exact single-shot calls, custom-header extraction/encoding and cancellation/deadline checks.

## Specification references

- [Streamable HTTP, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Versioning and compatibility, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [Streamable HTTP, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- [Lifecycle and capabilities, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)
