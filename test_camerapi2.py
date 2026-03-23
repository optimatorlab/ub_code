"""
Quick integration test for CameraPi2.
Run on the Pi with:
    source ~/venv/bin/activate
    python test_camerapi2.py
"""

import time
import sys

print("Importing ub_camera...")
try:
    from ub_camera import CameraPi2
    print("  OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

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

print()
print("Stream running. Open in browser:")
print("  http://192.168.0.105:8000")
print()
print("Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(2)
        # Print a basic frame/FPS status line every 2 seconds
        print(f"  capture fps: {cam.fps['capture'].actual}", flush=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    cam.shutdown()
    print("Done.")
