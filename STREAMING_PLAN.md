# ub_camera Streaming Enhancement Plan

## Background

`ub_camera` currently streams video as MJPEG over HTTPS using a pure-Python
`socketserver.ThreadingMixIn` + `http.server.HTTPServer` with a self-signed
SSL certificate bundled at `ub_camera/ssl/ca.crt`.

This plan adds two new output protocols — **WebSocket + JPEG** and **WebRTC** —
while keeping MJPEG fully intact for backward compatibility.

---

## Constraints and Design Principles

- **Backward compatible.** `startStream(port=8000)` continues to work exactly
  as today (MJPEG, no changes to existing callers).
- **Cross-platform.** Must work on Windows, Mac, and Linux. No Linux-specific
  dependencies (no Avahi, no GStreamer, no system-level cert tooling).
- **Open source friendly.** No dependency on any specific lab's domain,
  proxy, or infrastructure. Everything works out of the box.
- **Existing frame pipeline is untouched.** `frameDeque` (maxlen=1),
  `threading.Condition`, and `decorateFrame()` are shared by all three
  protocols — no changes to capture threads or CV processing.
- **Multi-client broadcast supported** for all three protocols.
- **Inter-instance (robot → computer) leg keeps MJPEG.** WebRTC input is
  not implemented (too complex, marginal benefit on LAN). MJPEG input via
  `cv2.VideoCapture(url)` continues to work as the inter-instance transport.

---

## Proposed API

```python
# Existing — unchanged
camera.startStream(port=8000)
camera.startStream(port=8000, protocol='mjpeg')   # explicit, same behavior

# New
camera.startStream(port=8001, protocol='websocket')
camera.startStream(port=8002, protocol='webrtc')

# Stop — unchanged
camera.stopStream()
```

The `protocol` parameter defaults to `'mjpeg'` so all existing code continues
to work without modification.

Multiple protocols can run simultaneously on different ports:

```python
camera.startStream(port=8000, protocol='mjpeg')      # backward-compat clients
camera.startStream(port=8001, protocol='websocket')  # lower-latency clients
camera.startStream(port=8002, protocol='webrtc')     # browser-facing / lowest latency
```

---

## Architecture Overview

```
frameDeque (maxlen=1)  +  threading.Condition   [unchanged]
        │
        ├── _thread_stream_mjpeg()      → StreamingServer (existing)
        ├── _thread_stream_websocket()  → WebSocketStreamingServer (new)
        └── _thread_stream_webrtc()     → WebRTCStreamingServer (new)
```

`startStream()` dispatches to the appropriate thread based on `protocol`.
All three backends read from the same `frameDeque` and wait on the same
`threading.Condition`, so they compose cleanly with each other and with all
existing CV processing (ArUco, YOLO, face detection, etc.).

---

## Step 1 — Refactor `startStream()` and `stopStream()`

**Current signatures:**
```python
def startStream(self, port):
def stopStream(self):
```

**Proposed signatures:**
```python
def startStream(self, port, protocol='mjpeg'):
def stopStream(self, protocol=None):   # None = stop all active protocols
```

`startStream()` will dispatch:
```python
if protocol == 'mjpeg':
    # existing _thread_stream() logic, renamed _thread_stream_mjpeg()
elif protocol == 'websocket':
    # new _thread_stream_websocket()
elif protocol == 'webrtc':
    # new _thread_stream_webrtc()
```

`keepStreaming` and `numStreams` remain scalars — only one protocol runs at
a time (see Q4 below). The active protocol is tracked in a new attribute:
```python
self.keepStreaming   = False
self.numStreams      = 0
self.activeProtocol = None   # 'mjpeg' | 'websocket' | 'webrtc'
```

`streamIncr()` requires no changes to its signature.

Calling `startStream()` while a stream is already active raises a
`RuntimeError` unless the caller passes `force=True`, which stops the
current stream before starting the new one:
```python
camera.startStream(port=8002, protocol='webrtc', force=True)
```

---

## Step 2 — WebSocket + JPEG Output

### New dependency
```
websockets>=12.0
```
Added as an optional dependency in `pyproject.toml`:
```toml
[project.optional-dependencies]
websocket = ["websockets>=12.0"]
```

### How it works

The WebSocket server broadcasts binary JPEG frames to all connected clients.
Each frame goes through the same path as MJPEG:
1. Copy from `frameDeque`
2. `decorateFrame()` applied
3. `cv2.imencode('.jpg', frame)` → bytes
4. Sent as binary WebSocket message to all connected clients

The server uses `wss://` (WebSocket Secure, TLS) so it can be embedded in
an `https://` page — same self-signed cert as today, same one-time browser
warning behavior, but `wss://` is required for mixed-content compliance.

### New class: `WebSocketStreamingServer`

```
class WebSocketStreamingServer:
    - Wraps asyncio + websockets.serve()
    - Maintains a set of connected client WebSocket objects
    - Runs a broadcaster coroutine that waits on frameDeque via asyncio.Queue
    - Applies SSL via ssl.SSLContext (same cert loading as existing code)
    - Runs in a daemon thread with its own asyncio event loop
      (isolates asyncio from the existing threading model)
```

### New thread: `_thread_stream_websocket(portNumber)`

Mirrors the structure of `_thread_stream_mjpeg()`:
- Creates the `WebSocketStreamingServer`
- Wraps it with SSL context (same cert, same code path)
- Runs `asyncio.run(server.serve_forever())` in the thread
- Respects `keepStreaming` flag

### IP allowlist / blocklist enforcement

Enforced on WebSocket connection open (equivalent to MJPEG's `do_GET` check).
If the connecting client's IP is blocked, the server closes the WebSocket
immediately after the handshake with a `1008 Policy Violation` close code.
The per-frame re-check present in the MJPEG handler is also replicated in the
broadcaster loop.

### Browser client

A companion HTML snippet (added to docs / README) shows how to receive the
stream in a browser:
```html
<canvas id="cam"></canvas>
<script>
  const ws = new WebSocket('wss://camera-host:8001');
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    const blob = new Blob([e.data], {type: 'image/jpeg'});
    createImageBitmap(blob).then(bmp => ctx.drawImage(bmp, 0, 0));
  };
</script>
```

### Latency improvement over MJPEG

- Eliminates multipart HTTP boundary overhead per frame
- Eliminates per-frame HTTP headers
- Reduces browser buffering (WebSocket is message-oriented, not a byte stream)
- Estimated improvement: **50–150ms** on a typical LAN/WiFi path

---

## Step 3 — WebRTC Output

### New dependencies
```
aiortc>=1.9.0
aiohttp>=3.9.0
```
Added as an optional dependency group:
```toml
[project.optional-dependencies]
webrtc = ["aiortc>=1.9.0", "aiohttp>=3.9.0"]
```

### How it works

WebRTC requires a two-phase setup:

**Phase 1 — Signaling (one-time per client connection):**
1. Browser fetches `GET /webrtc` → response depends on `signalingMode` (see below)
2. Browser POSTs an SDP offer to `POST /offer`
3. `aiohttp` server receives the offer, creates an `RTCPeerConnection` via
   `aiortc`, generates an SDP answer, returns it to the browser as JSON
4. Browser and server complete ICE negotiation

**Phase 2 — Media (ongoing):**
- Video flows directly via UDP (DTLS/SRTP) — never touches the signaling server
- `aiortc` `VideoStreamTrack` subclass pulls frames from `frameDeque` and
  yields them to the WebRTC peer
- Each connected client has its own `RTCPeerConnection` but all share the same
  `frameDeque` source

### New class: `CameraVideoTrack(VideoStreamTrack)`

```python
class CameraVideoTrack(VideoStreamTrack):
    def __init__(self, camObject):
        super().__init__()
        self.camObject = camObject

    async def recv(self):
        # Wait for next frame via asyncio-compatible wrapper around condition
        frame = await self._get_next_frame()
        # Convert numpy array → aiortc VideoFrame (YUV420p)
        # Apply decorateFrame() before conversion
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame
```

`aiortc` calls `recv()` repeatedly to pull frames. The key design challenge
is bridging the existing `threading.Condition` (synchronous) with `aiortc`'s
`asyncio` event loop. This is solved with `asyncio.get_event_loop().run_in_executor()`
to wait on the condition from an async context without blocking the event loop.

### Signaling mode — built-in HTML or JSON only

`startStream()` accepts a `signalingMode` parameter:

```python
camera.startStream(port=8002, protocol='webrtc', signalingMode='html')   # default
camera.startStream(port=8002, protocol='webrtc', signalingMode='json')
```

- **`'html'` (default):** `GET /webrtc` returns a self-contained HTML page
  (bundled in the package) with the full JavaScript signaling + `<video>`
  element. Students can open `https://camera-host:8002/webrtc` directly in a
  browser with no additional setup.
- **`'json'`:** `GET /webrtc` returns a minimal JSON descriptor
  (`{"offerUrl": "/offer"}`). The caller is responsible for their own UI.
  Intended for integration into existing web pages or custom frontends.

`POST /offer` behaves identically in both modes — it always returns a JSON
SDP answer. The mode only affects what `GET /webrtc` returns.

### New class: `WebRTCStreamingServer`

```
class WebRTCStreamingServer:
    - aiohttp Application with two routes:
        GET  /webrtc  → built-in HTML page (signalingMode='html')
                        OR JSON descriptor  (signalingMode='json')
        POST /offer   → handles SDP offer/answer exchange (always JSON)
    - Maintains a list of active RTCPeerConnection objects
    - Each POST /offer creates a new RTCPeerConnection + CameraVideoTrack
    - SSL via aiohttp's ssl_context parameter (same cert as existing code)
    - Runs in a daemon thread with its own asyncio event loop
```

### New thread: `_thread_stream_webrtc(portNumber)`

- Creates `WebRTCStreamingServer`
- Configures SSL context (same cert loading as existing code)
- Runs `aiohttp.web.run_app()` in the thread's asyncio event loop
- Respects `keepStreaming` flag
- On shutdown, closes all active `RTCPeerConnection` objects cleanly

### IP allowlist / blocklist enforcement

Enforced at `POST /offer` before any `RTCPeerConnection` is created. If the
requesting IP is blocked, the server returns HTTP 403 immediately. This is the
equivalent enforcement point to MJPEG's `do_GET` check.

### Signaling and the self-signed cert

The signaling page is served over `https://` using the same self-signed cert
as the MJPEG stream. Students still see the one-time browser warning for the
signaling endpoint. **The media stream itself (WebRTC DTLS) never triggers a
cert warning** — this is handled internally by the browser's WebRTC
implementation.

### Lab deployment with a real cert

For labs that want zero cert friction (e.g., using a subdomain with a
Let's Encrypt cert as a signaling proxy), the `sslPath` parameter already
supports custom cert paths. No code changes needed — swap in a real cert and
the warning disappears. The WebRTC media still flows directly
camera→browser, never touching the proxy.

### Latency improvement over MJPEG

- UDP transport eliminates TCP head-of-line blocking
- Browser-native decode (hardware-accelerated H.264/VP8)
- No buffering overhead
- Estimated improvement: **100–300ms** over MJPEG, **50–150ms** over WebSocket

---

## Step 4 — `pyproject.toml` Changes

```toml
[project.optional-dependencies]
yolo      = ["ultralytics>=8.3.0"]
ros       = ["rospy", "cv-bridge", "sensor-msgs"]
websocket = ["websockets>=12.0"]
webrtc    = ["aiortc>=1.9.0", "aiohttp>=3.9.0"]
all       = ["ultralytics>=8.3.0", "rospy", "cv-bridge", "sensor-msgs",
             "websockets>=12.0", "aiortc>=1.9.0", "aiohttp>=3.9.0"]
```

Both `websockets` and `aiortc` are imported lazily (inside
`_thread_stream_websocket` / `_thread_stream_webrtc`) with a clear error
message if not installed:

```python
try:
    import websockets
except ImportError:
    raise ImportError("WebSocket streaming requires 'websockets'. "
                      "Install with: pip install ub-code[websocket]")
```

This preserves the existing install experience — users who only need MJPEG
install nothing extra.

---

## Step 5 — Documentation Updates

- Update module docstring to list all three streaming protocols
- Update `startStream()` docstring with `protocol` parameter
- Add a "Streaming Protocols" section to `README.md` with:
  - Comparison table (latency, cert behavior, dependencies, browser support)
  - Example code for each protocol
  - Browser client HTML snippets for WebSocket and WebRTC
  - Note on using `sslPath` with a real cert for zero-warning deployment

---

## What Is NOT in This Plan

| Item | Reason |
|---|---|
| WebRTC **input** (`CameraWebRTC`) | Requires aiortc peer + SDP negotiation; no URL-based interface possible; marginal benefit on LAN vs. MJPEG |
| WebSocket **input** (`CameraWebSocket`) | Deferred — inter-instance MJPEG input works well on LAN; add later if latency measurements justify it |
| mkcert / local CA tooling | Requires per-machine root CA install; not appropriate for students on unmanaged machines |
| Dropping SSL entirely | HTTPS pages cannot embed `http://` or `ws://` content (mixed-content blocking) |
| RTSP output | No clear benefit over the three protocols above for this use case |

---

## Implementation Order

| Step | Scope | Risk | Effort |
|---|---|---|---|
| 1. Refactor `startStream()` | `Camera.__init__`, `startStream`, `stopStream`, `streamIncr` | Low — additive, no behavior change for MJPEG | Small |
| 2. WebSocket output | New `WebSocketStreamingServer`, `_thread_stream_websocket` | Low — well-understood library | Medium |
| 3. WebRTC output | New `CameraVideoTrack`, `WebRTCStreamingServer`, `_thread_stream_webrtc` | Medium — asyncio/threading bridge, aiortc API | Large |
| 4. `pyproject.toml` | Add optional deps, lazy imports with helpful errors | Low | Small |
| 5. Docs | Module docstring, README | Low | Small |

Steps 1 and 2 are independent of WebRTC and deliver value immediately.
Step 3 builds on Step 1 but is otherwise independent of Step 2.

---

## Design Decisions (Resolved)

1. **`stopStream()` with no argument** — stops the active protocol (only one
   runs at a time). No `protocol` argument needed.

2. **WebRTC signaling page** — both modes supported via `signalingMode`:
   - `'html'` (default): `GET /webrtc` returns a self-contained HTML+JS page
     bundled in the package; students open it directly in a browser.
   - `'json'`: `GET /webrtc` returns a JSON descriptor for custom UI integration.
   - `POST /offer` always returns a JSON SDP answer regardless of mode.

3. **IP allowlist / blocklist** — enforced consistently across all protocols:
   - MJPEG: in `StreamingHandler.do_GET()` (existing)
   - WebSocket: on connection open, close with `1008 Policy Violation` if blocked
   - WebRTC: at `POST /offer`, return HTTP 403 if blocked

4. **Simultaneous protocols** — one active protocol at a time. Calling
   `startStream()` while a stream is already active raises `RuntimeError`
   unless `force=True` is passed, which stops the current stream first.
