# Spec: `CameraPi2` Class

## Overview

Port `CameraPi` (picamera) to `CameraPi2` (picamera2 v0.3.34+).
Both classes live side-by-side in `ub_camera/__init__.py`, placed immediately after `CameraPi`.
`CameraPi2` inherits from `Camera` with an identical public API to `CameraPi`.

---

## Key Differences: picamera → picamera2

| Concern | CameraPi (picamera) | CameraPi2 (picamera2) |
|---|---|---|
| Frame delivery | Push callback `write(buf)` | Pull loop — `capture_array("main")` in background thread |
| Color format | BGR natively | Configure as `"BGR888"` to avoid conversion |
| Zoom | `cap.zoom = (x, y, w, h)` — normalized 0–1 | `set_controls({"ScalerCrop": (x, y, w, h)})` — pixel coords, queried from `camera_properties["PixelArraySize"]` |
| Framerate | `cap.framerate_delta` | `set_controls({"FrameDurationLimits": (us, us)})` |
| Resolution change | `stop_recording` / change / `start_recording` | Stop capture thread → `cap.stop()` → reconfigure → `cap.start()` → restart thread |
| Preview | Implicit | Headless — no preview call needed (null is default when no display) |
| Import | `import picamera` | `from picamera2 import Picamera2` |

---

## New Internals

Two new instance attributes (not in `CameraPi`):

- `self._capture_thread` — `threading.Thread` running the frame-pull loop
- `self._capture_running` — `bool` flag; set `False` to signal the thread to exit

---

## Method-by-Method Plan

### `__init__`

Identical signature to `CameraPi.__init__`.

```
try:
    from picamera2 import Picamera2
    self.Picamera2 = Picamera2
except Exception as e:
    print(f'Failed to init CameraPi2: {e}')

super().__init__(...)

self.cap = None
self._capture_thread = None
self._capture_running = False
```

---

### `start(assetID, res_rows, res_cols, framerate, startStream, port, protocol, imgTopic, compImgTopic)`

Identical signature to `CameraPi.start()`.

Steps:
1. Resolve defaults via `defaultFromNone` (same as CameraPi).
2. Instantiate `Picamera2()`.
3. Compute `frame_duration_us = int(1e6 / framerate)`.
4. Build video config:
   ```python
   config = self.cap.create_video_configuration(
       main={"format": "BGR888", "size": (res_cols, res_rows)}
   )
   self.cap.configure(config)
   ```
5. `self.cap.start()`
6. Set framerate controls post-start:
   ```python
   self.cap.set_controls({"FrameDurationLimits": (frame_duration_us, frame_duration_us)})
   ```
7. Read back actual resolution and framerate from `cap.camera_configuration()["main"]["size"]`
   and `cap.camera_controls["FrameDurationLimits"]`; call `updateResolution` / `updateFramerate`.
   - Note: if reading back framerate is unreliable, fall back to the requested value.
8. Set `self.camOn = True`.
9. Start capture thread (see `_startCaptureThread`).
10. Optionally `startStream`, `startROStopic` — identical to `CameraPi`.
11. `reachback_pubCamStatus()`.

Wrap entire body in `try/except`, log error at `SEVERITY_ERROR`.

---

### `_startCaptureThread` *(new private helper)*

```python
def _startCaptureThread(self):
    self._capture_running = True
    self._capture_thread = threading.Thread(target=self._captureLoop, daemon=True)
    self._capture_thread.start()
```

---

### `_captureLoop` *(new private method — replaces `write()`)*

```python
def _captureLoop(self):
    while self._capture_running:
        try:
            frame = self.cap.capture_array("main")   # blocks until next frame
            self.frameDeque.append(frame)
            self.announceCondition()
            self.calcFramerate(self.fps['capture'], 'capture')
        except Exception as e:
            self.logger.log(f'Error in CameraPi2 capture loop: {e}', severity=ub_utils.SEVERITY_ERROR)
```

Frame is already BGR (configured as `BGR888`) — no conversion needed.

---

### `_stopCaptureThread` *(new private helper)*

```python
def _stopCaptureThread(self, timeout=3.0):
    self._capture_running = False
    if self._capture_thread is not None:
        self._capture_thread.join(timeout=timeout)
        self._capture_thread = None
```

---

### `stop`

```python
def stop(self):
    try:
        self.camOn = False
        self._stopCaptureThread()
        self.cap.stop()
        self.stopStream()
    except Exception as e:
        raise Exception(f'Error in camera stop: {e}')
```

---

### `shutdown`

```python
def shutdown(self):
    try:
        if self.cap:
            self.stop()
            self.cap.close()
            time.sleep(STREAM_MAX_WAIT_TIME_SEC + 1)
    except Exception as e:
        self.logger.log(f'Error in camera shutdown: {e}', severity=ub_utils.SEVERITY_ERROR)
```

Identical structure to `CameraPi.shutdown`.

---

### `_changeFramerate(req_framerate)`

```python
if req_framerate == self.fps_target:
    return (True, '')

if self.fpsMin <= req_framerate <= self.fpsMax:
    frame_duration_us = int(1e6 / req_framerate)
    self.cap.set_controls({"FrameDurationLimits": (frame_duration_us, frame_duration_us)})
    self.updateFramerate(req_framerate)
    return (True, '')
else:
    return (False, 'picam2 framerate is at limit')
```

Wrap in `try/except`, return `(False, message)` on error.

---

### `_changeResolution(req_height, req_width)`

```python
current_size = self.cap.camera_configuration()["main"]["size"]  # (width, height)
if current_size == (req_width, req_height):
    return (False, f'picam2 resolution is already {req_width}x{req_height}.')

self._stopCaptureThread()
self.cap.stop()

config = self.cap.create_video_configuration(
    main={"format": "BGR888", "size": (req_width, req_height)}
)
self.cap.configure(config)
self.cap.start()
self.cap.set_controls({"FrameDurationLimits": (int(1e6 / self.fps_target), int(1e6 / self.fps_target))})

self.updateResolution(req_height, req_width)
self._startCaptureThread()
return (True, '')
```

Wrap in `try/except`, return `(False, message)` on error.

---

### `changeResolutionFramerate(res_rows, res_cols, framerate)`

Identical logic to `CameraPi.changeResolutionFramerate` — resolves defaults, calls `_changeFramerate` then `_changeResolution`, raises combined error message if either fails.

---

### `changeZoom(zoomLevel)`

```python
sensor_w, sensor_h = self.cap.camera_properties["PixelArraySize"]
crop_w = int(sensor_w / zoomLevel)
crop_h = int(sensor_h / zoomLevel)
crop_x = (sensor_w - crop_w) // 2
crop_y = (sensor_h - crop_h) // 2
self.cap.set_controls({"ScalerCrop": (crop_x, crop_y, crop_w, crop_h)})
self.updateZoom(zoomLevel)
```

Wrap in `try/except`, log at `SEVERITY_ERROR` on failure (same pattern as `CameraPi.changeZoom`).

---

### `write` — **not implemented**

The `write()` callback is a picamera-specific interface. It has no equivalent in picamera2. Do not include this method.

---

### `takePhotoLocal`

Inherit from base `Camera` class (grabs latest frame from `frameDeque`). No override needed.

---

### ROS

`startROStopic(imgTopic, compImgTopic)` is inherited from `Camera` and called from `start()` identically to `CameraPi`. No override needed.

---

## Hardware / Platform Notes

- Targets: Pi 3B, 4, 5, CM5 running Raspberry Pi OS Trixie (Debian 13).
- Camera modules: v2 (IMX219), v3 (IMX708), HQ (IMX477).
- picamera2 version: 0.3.34-1 (from apt).
- `PixelArraySize` queried at runtime — handles all sensor variants automatically.
- `ScalerCrop` center-crops the sensor; works identically across all supported modules.

---

## What is NOT Changing

- Public API / method signatures — drop-in compatible with `CameraPi`.
- `frameDeque` population — streaming (MJPEG/WebSocket/WebRTC) is unaffected.
- Error handling style — mirrors existing `CameraPi` pattern.
- `assetID` parameter — accepted, not used (same as `CameraPi`).
- `device` and `apiPref` parameters — accepted, not used (same as `CameraPi`).

---

## Open Questions / FIXMEs

1. **Framerate readback**: picamera2's `camera_controls["FrameDurationLimits"]` returns the hardware range, not the currently-set value. After `start()`, the set framerate may not be easily readable back. Suggest storing `req_framerate` as truth and skipping the readback, or verifying with a known-good approach on the target hardware before implementing.
2. **Pi 3B performance**: The pull loop with `capture_array()` on a Pi 3B may have higher CPU overhead than picamera's push callback. Flag for profiling during initial integration testing.
3. **`FIXME` in CameraPi re: `device`**: The original class has an open FIXME about whether `device` is ever used. `CameraPi2` inherits the same ambiguity — leaving as-is.
