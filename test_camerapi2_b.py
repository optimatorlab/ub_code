"""
CameraPi2 test: grayscale stream.
Stops the internal capture thread and runs a custom frame loop that
converts each frame to grayscale before pushing it into the deque.

Run on the Pi with:
    source ~/venv/bin/activate
    python test_camerapi2_b.py
"""

import time
import sys
import cv2
import ub_utils
from ub_camera import CameraPi2


print("Creating CameraPi2 instance...")
try:
    cam = CameraPi2()
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("Starting camera + MJPEG stream on port 8000...")
try:
    cam.start(res_rows=480, res_cols=640, framerate=30, startStream=True, port=8000, protocol='mjpeg')
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    cam.shutdown()
    sys.exit(1)

# Hand off frame delivery to our own loop
cam._stopCaptureThread()

print()
print("Stream running. Open in browser:")
print("  https://192.168.0.105:8000/stream.mjpg")
print()
print("Press Ctrl+C to stop.")

try:
    while True:
        frame = cam.cap.capture_array("main")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cam.frameDeque.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        cam.announceCondition()
        cam.calcFramerate(cam.fps['capture'], 'capture')

        # Status line every ~2 seconds
        if cam.fps['capture'].actual > 0 and cam.fps['capture'].numFrames % (cam.fps['capture'].actual * 2) == 0:
            print(f"  capture fps: {cam.fps['capture'].actual}", flush=True)

except KeyboardInterrupt:
    print("\nShutting down...")
    cam.shutdown()
    print("Done.")
