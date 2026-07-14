# Job Progress Stream — SSE message contract (v2)

Replaces v1's `setInterval(jobstatus, 5000)` JSONP polling. The client opens **one**
`EventSource` per submitted job and the server pushes state changes and live MCX log lines
as they happen. Eliminates client polling; combined with the worker→API completion POST
(§2.5-1), it also eliminates the NFS-visibility lag.

## Endpoint

```
GET /jobs/{id}/stream?token=<capability-token>     Accept: text/event-stream
```

Standard SSE framing (`data:` lines, `\n\n` terminated). Each event `data:` payload is a
single-line JSON object. Named events use the SSE `event:` field.

## Event types

| `event:` | when | payload fields |
|---|---|---|
| `status` | job state transition | `status`, `jobid`, `queuePos?`, `gpu?` |
| `log`    | new MCX stdout/log line(s) | `jobid`, `line` (ANSI already stripped server-side) |
| `progress` | periodic during run | `jobid`, `percent` (0–100), `photons?`, `speed?` |
| `complete` | output stored, ready | `jobid`, `outputHash`, `hasDetphoton` (bool), `runtime` |
| `error`  | failure/invalid/killed | `jobid`, `status`, `message` |

`status` values (superset of v1's map): `queued`, `initiated`, `created`, `running`,
`writing`, `completed`, `cached`, `failed`, `invalid`, `cancelled`, `killed`.

## Examples

```
event: status
data: {"jobid":"9F2A…","status":"queued","queuePos":2}

event: status
data: {"jobid":"9F2A…","status":"running","gpu":"RTX2080S#3"}

event: log
data: {"jobid":"9F2A…","line":"MCX simulation speed: 6775.31 photon/ms"}

event: progress
data: {"jobid":"9F2A…","percent":80,"speed":6612}

event: complete
data: {"jobid":"9F2A…","outputHash":"sha256/ab34…","hasDetphoton":true,"runtime":5.12}
```

## Client & server rules

- On `complete`, the client `GET`s `/jobs/{id}/output` (JNIfTI, reassembled) and renders;
  if `hasDetphoton`, `/jobs/{id}/detphoton` is available. The client then **closes** the
  stream.
- On `error`, the server sends the event then closes the stream.
- **Heartbeat:** server sends a `: keep-alive` comment every ~15 s so proxies don't drop
  idle connections.
- **Reconnect:** SSE `id:` carries the last event sequence number; on reconnect the client
  sends `Last-Event-ID` and the server replays from there (or returns the terminal state
  immediately if the job already finished — covers the cached-result case).
- **Auth:** the per-job capability token (returned by `POST /jobs`) is required; a bad/absent
  token → `403` before the stream opens.
- Cancellation is a separate `DELETE /jobs/{id}` (REST), not an SSE upstream message —
  SSE is one-way, which is why WebSocket is unnecessary here.
