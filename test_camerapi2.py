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

# Grab one raw frame directly from picamera2 for diagnostics (bypasses _captureLoop)
import numpy as np
time.sleep(1)  # let capture thread warm up
raw = cam.cap.capture_array("main")
print(f"  frame shape: {raw.shape}, dtype: {raw.dtype}")
print(f"  top-left pixel (should be roughly sky/ceiling): {raw[0,0,:]}")
print(f"  Interpretation: channel order is [0]={raw[0,0,0]} [1]={raw[0,0,1]} [2]={raw[0,0,2]}")
print()
print("Stream running. Open in browser:")
print("  https://192.168.0.105:8000/stream.mjpg")
print()
print("Press Ctrl+C to stop.")
print("(Point camera at something with a distinct red or blue object to verify colors)")

try:
    while True:
        time.sleep(2)
        # Print a basic frame/FPS status line every 2 seconds
        print(f"  capture fps: {cam.fps['capture'].actual}", flush=True)
except KeyboardInterrupt:
    print("\nShutting down...")
    cam.shutdown()
    print("Done.")
