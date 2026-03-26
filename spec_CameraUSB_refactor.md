# CameraUSB Refactor Plan: Explicit Capture Thread Pattern

## Motivation

`CameraUSB._thread_capture()` currently does two things in one method:
1. Opens and configures `cv2.VideoCapture` (hardware setup)
2. Runs the frame-reading loop until `self.camOn` is False

`CameraPi2` separates these responsibilities cleanly: hardware setup happens synchronously in `start()`, and frame delivery runs in `_captureLoop()`, managed by `_startCaptureThread()` / `_stopCaptureThread()`. This refactor brings `CameraUSB` in line with that pattern.

---

## Additional Goal: User-Facing Frame Interception

The refactor is an opportunity to expose a `frameProcessor` hook — a user-assignable callable that runs on every captured frame. This enables on-the-fly CV customization without subclassing:

```python
cam = CameraUSB(device='/dev/video0')

def my_pipeline(frame):
    frame = apply_color_filter(frame)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame           # return edited frame → it streams
    # return None          # return None → frame is dropped (not streamed)

cam.frameProcessor = my_pipeline
cam.start(startStream=True, port=8000)
```

- **Return a frame** → it is appended to `frameDeque` and streamed.
- **Return `None`** → the frame is discarded (not streamed, not published).
- **`frameProcessor = None`** (default) → pass-through, unchanged behavior.

The hook fires *after* `zoomFunction` is applied, so the user always receives a correctly-zoomed frame.

---

## Summary of Changes

| What | Old | New |
|---|---|---|
| Hardware setup | Inside `_thread_capture()` thread | In `start()`, synchronously |
| Frame loop | `_thread_capture()` (setup + loop together) | `_captureLoop()` (loop only) |
| Thread lifecycle | No explicit start/stop helpers | `_startCaptureThread()` / `_stopCaptureThread()` |
| Thread stop signal | `self.camOn` flag + `cap.isOpened()` | Dedicated `self._capture_running` flag |
| `cap.release()` | Inside `_thread_capture()` when loop exits | In `stop()`, explicitly |
| Frame interception | Not supported | Optional `self.frameProcessor` hook |
| `stop(stopStream)` | Preserved | Preserved (required by `changeResolutionFramerate`) |

---

## New Attributes (added in `__init__`)

```python
self._capture_thread   = None   # threading.Thread running _captureLoop
self._capture_running  = False  # flag to signal loop to stop
self.frameProcessor    = None   # optional callable(frame) -> frame | None
```

---

## New / Modified Methods

### `_startCaptureThread()`  *(new)*

```python
def _startCaptureThread(self):
    """Start the background frame capture thread."""
    self._capture_running = True
    self._capture_thread = threading.Thread(target=self._captureLoop, daemon=True)
    self._capture_thread.start()
```

Identical in structure to `CameraPi2._startCaptureThread()`.

---

### `_stopCaptureThread(timeout=3.0)`  *(new)*

```python
def _stopCaptureThread(self, timeout=3.0):
    """Signal the capture thread to stop and wait for it to finish."""
    self._capture_running = False
    if self._capture_thread is not None:
        self._capture_thread.join(timeout=timeout)
        self._capture_thread = None
```

Identical in structure to `CameraPi2._stopCaptureThread()`.

---

### `_captureLoop()`  *(new — replaces the loop body of `_thread_capture`)*

```python
def _captureLoop(self):
    """Background thread: pull frames from cv2.VideoCapture and populate frameDeque.

    Runs until _capture_running is False or cap stops being opened.
    Applies zoomFunction per frame, then calls frameProcessor (if set).
    If frameProcessor returns None the frame is dropped (not streamed).
    """
    while self._capture_running:
        try:
            if not self.cap.isOpened():
                self.logger.log('CameraUSB: VideoCapture closed unexpectedly', severity=ub_utils.SEVERITY_ERROR)
                break

            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = self.zoomFunction(frame)

            if self.frameProcessor is not None:
                frame = self.frameProcessor(frame)
                if frame is None:
                    continue   # user chose to drop this frame

            self.frameDeque.append(frame)
            self.announceCondition()
            self.calcFramerate(self.fps['capture'], 'capture')

        except Exception as e:
            self.logger.log(f'Error in CameraUSB capture loop: {e}', severity=ub_utils.SEVERITY_ERROR)

    self.camOn = False  # mirror existing behaviour: camOn=False when loop exits
```

Key differences from the old `_thread_capture` loop:
- Uses `self._capture_running` (not `self.camOn`) as the stop signal, matching `CameraPi2`.
- `cap.release()` is **not** called here; `stop()` owns it.
- `frameProcessor` hook inserted between zoom and deque append.

---

### `start()`  *(modified — hardware setup moved here)*

```python
def start(self, assetID=None, res_rows=None, res_cols=None, framerate=None,
          device=None, apiPref=None, startStream=False, port=None,
          protocol='mjpeg', imgTopic=None, compImgTopic=None):
    try:
        self.res_rows  = self.defaultFromNone(res_rows,  self.res_rows,  int)
        self.res_cols  = self.defaultFromNone(res_cols,  self.res_cols,  int)
        self.framerate = self.defaultFromNone(framerate, self.fps_target, int)
        self.device    = self.defaultFromNone(device,    self.device)
        self.apiPref   = self.defaultFromNone(apiPref,   self.apiPref)
        self.port      = self.defaultFromNone(port,      self.outputPort)

        # --- Hardware setup (was inside _thread_capture) ---
        if self.apiPref is None:
            self.cap = cv2.VideoCapture(self.device)
        else:
            params = [cv2.CAP_PROP_FRAME_WIDTH,  int(self.res_cols),
                      cv2.CAP_PROP_FRAME_HEIGHT, int(self.res_rows),
                      cv2.CAP_PROP_FPS,          int(self.framerate)]
            if self.fourcc is not None:
                fourcc_code = cv2.VideoWriter.fourcc(*self.fourcc)
                params.extend([cv2.CAP_PROP_FOURCC, fourcc_code])
            self.cap = cv2.VideoCapture(self.device, self.apiPref, params=params)

        if not self.cap.isOpened():
            raise Exception(f'cv2.VideoCapture failed to open: {self.device}')

        # Read back what the driver actually configured
        self.updateResolution(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                              self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.updateFramerate(self.cap.get(cv2.CAP_PROP_FPS))

        self.camOn = True
        self._startCaptureThread()
        # --- End hardware setup ---

        if startStream:
            if self.port is None:
                raise Exception('cannot stream when port is None')
            self.startStream(self.port, protocol=protocol)

        if (imgTopic is not None) or (compImgTopic is not None):
            self.startROStopic(imgTopic=imgTopic, compImgTopic=compImgTopic)

        self.reachback_pubCamStatus()

    except Exception as e:
        self.logger.log(f'Error in camera start: {e}', severity=ub_utils.SEVERITY_ERROR)
```

Benefit: if `VideoCapture` fails to open, the exception is logged in `start()` with a clear message, not silently inside a background thread.

---

### `stop(stopStream=True)`  *(modified — now owns `cap.release()`)*

```python
def stop(self, stopStream=True):
    """Stop the capture thread and release VideoCapture.

    Args:
        stopStream (bool): Whether to also stop the streaming server.
            Set False when changing resolution/framerate mid-stream.
    """
    self.camOn = False
    self._stopCaptureThread()
    if self.cap is not None:
        self.cap.release()
        self.cap = None
    if stopStream:
        self.stopStream()
```

`stopStream` parameter is preserved — `changeResolutionFramerate` passes `stopStream=False`.

---

### `changeResolutionFramerate()`  *(minor cleanup)*

The comparison logic changes slightly because after the refactor, `self.cap` may be `None` (if stop() was already called). Use stored attributes instead of live `cap.get()` queries for the "is change needed?" check:

```python
def changeResolutionFramerate(self, res_rows=None, res_cols=None, framerate=None):
    try:
        res_rows  = self.defaultFromNone(res_rows,  self.res_rows,   int)
        res_cols  = self.defaultFromNone(res_cols,  self.res_cols,   int)
        framerate = self.defaultFromNone(framerate, self.fps_target, int)

        if hasattr(self, 'fpsMin') and hasattr(self, 'fpsMax'):
            if not (self.fpsMin <= framerate <= self.fpsMax):
                raise Exception(f'framerate {framerate} outside [{self.fpsMin},{self.fpsMax}] bounds.')

        if (framerate != self.fps_target or
                res_rows != self.res_rows or
                res_cols != self.res_cols):

            self.stop(stopStream=False)
            time.sleep(1)
            self.start(res_rows=res_rows, res_cols=res_cols, framerate=framerate)

        # No redundant updateResolution/updateFramerate needed here —
        # start() already reads back actuals from the driver.

        fourccText = self.fourcc2text()
        self.logger.log(f'rows: {self.res_rows}, cols: {self.res_cols}, framerate: {framerate}',
                        severity=ub_utils.SEVERITY_DEBUG)

    except Exception as e:
        self.logger.log(f'Failed to change to {res_rows} rows, {res_cols} cols, {framerate} framerate: {e}',
                        severity=ub_utils.SEVERITY_ERROR)
```

Removed: the two trailing `updateResolution`/`updateFramerate` calls that queried `self.cap.get()` — `start()` now does this synchronously and correctly.

---

### `_thread_capture()`  *(deleted)*

The method is replaced entirely by `_startCaptureThread()`, `_stopCaptureThread()`, and `_captureLoop()`. No external code calls `_thread_capture()` directly (it was only used by `start()`), so it can be removed.

---

## Risk Checklist

| Risk | Mitigation |
|---|---|
| RTSP / HTTP sources (apiPref=None) | Handled by same `if apiPref is None` branch, now in `start()`. No logic change. |
| USB / V4L2 sources with fourcc | fourcc setup preserved verbatim, moved to `start()`. |
| `zoomFunction` per frame | Preserved in `_captureLoop()` as first transform before frameProcessor. |
| `cap.release()` ownership | Moved to `stop()` explicitly; removed from loop body. `self.cap = None` after release prevents double-release. |
| `stop(stopStream=False)` for `changeResolutionFramerate` | Parameter retained in new `stop()` signature. |
| Stream-alive during resolution change | `stop(stopStream=False)` → `start()` flow unchanged. Stream server is never stopped. |
| Open failure silent in thread | Now raises/logs in `start()` before thread is started. |
| `camOn` semantics | Set True in `start()`, False in `stop()` and at end of `_captureLoop()` (if cap closes unexpectedly). |
| `frameProcessor` returning None | `continue` in loop skips deque append and announceCondition — frame not streamed and not published to ROS. |

---

## What the refactor does NOT change

- Public API: `start()`, `stop()`, `shutdown()`, `changeResolutionFramerate()`, `changeZoom()` signatures are all preserved.
- `fourcc2text()` — unchanged.
- Streaming server lifecycle — unchanged.
- ROS topic publishing — unchanged.
- `zoomFunction` behavior — unchanged.
- `paramDict` handling / `defaultFromNone` logic — unchanged.

---

## Files to modify

- `ub_camera/__init__.py` — `CameraUSB` class only (lines ~4593–4982).

No other files need changes.

---

## Implementation Summary

| | Before | After |
|---|---|---|
| `__init__` | `cap = None` | + `_capture_thread`, `_capture_running`, `frameProcessor` |
| `_thread_capture()` | Setup + frame loop combined | **Deleted** |
| `_startCaptureThread()` | — | New |
| `_stopCaptureThread()` | — | New |
| `_captureLoop()` | — | New — includes `frameProcessor` hook |
| `start()` | Spawned thread, hardware setup inside thread | Opens/configures `VideoCapture` synchronously, then calls `_startCaptureThread()` |
| `stop()` | Set `camOn=False`, optionally `stopStream()` | + `_stopCaptureThread()` + `cap.release()` |
| `changeResolutionFramerate()` | Compared via `cap.get()` (unsafe post-stop) | Compares via `self.res_rows/fps_target`; removed redundant trailing `updateResolution/Framerate` calls; fixed fpsMax typo in error message |
