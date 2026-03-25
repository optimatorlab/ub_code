"""
CameraPi2 test: grayscale stream.
Subclasses CameraPi2 to override _captureLoop with grayscale conversion.

Run on the Pi with:
    source ~/venv/bin/activate
    python test_camerapi2_b.py
"""

import time
import sys
import cv2
import ub_utils
from ub_camera import CameraPi2


class CameraPi2Gray(CameraPi2):
    """CameraPi2 with grayscale conversion applied before frames enter the deque."""

    def _captureLoop(self):
        while self._capture_running:
            try:
                frame = self.cap.capture_array("main")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frameDeque.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                self.announceCondition()
                self.calcFramerate(self.fps['capture'], 'capture')
            except Exception as e:
                self.logger.log(f'Error in CameraPi2Gray capture loop: {e}', severity=ub_utils.SEVERITY_ERROR)


print("Creating CameraPi2Gray instance...")
try:
    cam = CameraPi2Gray()
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

print()
print("Stream running. Open in browser:")
print("  https://192.168.0.105:8000/stream.mjpg")
print()
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(2)
        print(f"  capture fps: {cam.fps['capture'].actual}", flush=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    cam.shutdown()
    print("Done.")
