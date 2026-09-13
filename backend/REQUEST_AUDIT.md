# Ollama request capture

After restarting the backend, local and cloud Ollama chat requests are recorded
in `logs/ollama_requests.jsonl` at the repository root. Each line contains the UTC
timestamp, benchmark mode, call type, destination host/path, and serialized SDK
body fields including `messages`, `think`, and `options`. The system prompt is
the entry in `payload.messages` whose `role` is `system`.

Capture runs in an HTTPX request hook after SDK serialization and before sending.
It establishes what the client attempts to send, not independently what the
server received or how the provider applied the system prompt. A captured request
can still fail during network transport.

Headers and API credentials are not recorded. Prompt and problem text are
recorded. Capture failures produce a warning without aborting generation.
Existing requests and historical results cannot be captured retroactively.
