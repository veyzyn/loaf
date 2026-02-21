# loaf rpc protocol (v1)

`loaf rpc` starts a JSON-RPC 2.0 server over stdio (NDJSON framing).

- transport: stdio
- framing: one JSON object per line
- protocol: JSON-RPC 2.0
- batches: not supported in v1

## startup

```bash
npm run rpc
# or
npm run dev -- rpc
```

## handshake

request:

```json
{"jsonrpc":"2.0","id":1,"method":"rpc.handshake","params":{"client_name":"demo","client_version":"0.1.0","protocol_version":"1.0.0"}}
```

response:

```json
{"jsonrpc":"2.0","id":1,"result":{"protocol_version":"1.0.0","server_name":"loaf","capabilities":{"events":true,"command_execute":true,"multi_session":true,"image_inputs":["path","data_url"]},"methods":["..."]}}
```

## methods

- `rpc.handshake`
- `system.ping`
- `system.shutdown`
- `state.get`
- `command.execute`
- `auth.status`
- `auth.connect.openai`
- `auth.connect.antigravity`
- `auth.set.openrouter_key`
- `auth.set.exa_key`
- `onboarding.status`
- `onboarding.complete`
- `model.list`
- `model.select`
- `model.openrouter.providers`
- `limits.get`
- `history.list`
- `history.get`
- `history.clear_session`
- `skills.list`
- `tools.list`
- `session.create`
- `session.get`
- `session.send`
- `session.steer`
- `session.interrupt`
- `session.queue.list`
- `session.queue.clear`
- `debug.set`

## events

server emits notifications with method `event`:

```json
{"jsonrpc":"2.0","method":"event","params":{"type":"session.status","timestamp":"2026-01-01T00:00:00.000Z","payload":{"session_id":"...","pending":true,"status_label":"thinking..."}}}
```

event `type` values:

- `session.status`
- `session.message.appended`
- `session.stream.chunk`
- `session.tool.call.started`
- `session.tool.call.completed`
- `session.tool.results`
- `session.completed`
- `session.interrupted`
- `session.error`
- `auth.flow.started`
- `auth.flow.url`
- `auth.flow.device_code`
- `auth.flow.completed`
- `auth.flow.failed`
- `state.changed`

## oauth in rpc mode

in RPC mode, loaf emits auth URLs/device-code details as events and does not auto-open browser windows.

## stability

protocol `1.0.0` is additive-first: new methods/events may be added without breaking existing method names/semantics.
