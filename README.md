# ub-code: Unified Camera Interface and Utilities

This package provides comprehensive camera interfaces (`ub_camera`) and utility functions (`ub_utils`) from Optimator Lab.

## Installation

### Step 0: Get the Code

First, clone or download the repository:

```bash
# Clone the repository
git clone https://github.com/optimatorlab/ub_code.git
cd ub_code
```

Or download the ZIP file from [https://github.com/optimatorlab/ub_code](https://github.com/optimatorlab/ub_code) and extract it.

**IMPORTANT:** Follow these steps **in order** to avoid opencv dependency conflicts.

### Step 1: Install Core Dependencies

The installation order is critical because `ultralytics` may install `opencv-python`, but we need `opencv-contrib-python` for ArUco marker support.

```bash
# Install numpy
pip install "numpy>=2.4.1"

# Install ultralytics (for YOLO object detection)
pip install "ultralytics>=8.4.7"

# Uninstall opencv-python (ultralytics may have installed this)
pip uninstall opencv-python
# Answer 'y' if prompted

# Uninstall opencv-contrib-python (in case it's already installed)
pip uninstall opencv-contrib-python
# Answer 'y' if prompted (you may get a warning that it's not installed - that's OK)

# Install opencv-contrib-python (required for ArUco markers, face detection, etc.)
pip install "opencv-contrib-python>=4.13.0.90"

# Install pyzbar (for barcode/QR code detection)
pip install "pyzbar>=0.1.9"
```

### Step 2: Install ub-code Package

Make sure you're in the `ub_code` directory, then install:

**Development Mode (Recommended for active development):**
```bash
pip install -e .
```

**Standard Installation:**
```bash
pip install .
```

**With ROS support (optional):**
```bash
pip install -e ".[ros]"
```

After installation, you can import the modules from anywhere on your machine:
```python
import ub_camera
import ub_utils
```

### Why This Order Matters

- **ultralytics** (YOLO) depends on opencv-python, but we need opencv-contrib-python for ArUco markers
- Both opencv-python and opencv-contrib-python cannot be installed simultaneously
- opencv-contrib-python includes all functionality of opencv-python plus additional modules

## Using Custom SSL Certificates

The package includes self-signed SSL certificates for HTTPS streaming (useful for development/testing). To use your own SSL certificates instead:

```python
# Pass the sslPath parameter when initializing your camera
camera = ub_camera.CameraUSB(
    paramDict={'res_rows': 480, 'res_cols': 640, 'fps_target': 30},
    sslPath='/path/to/your/ssl/directory'
)
```

Your SSL directory should contain:
- `ca.crt` - SSL certificate file
- `ca.key` - SSL private key file

If you don't specify `sslPath`, the package will automatically use the bundled certificates located in the `ub_camera/ssl/` directory.

---

# Introduction to the `ub_camera.py` module

This document describes some basic functionality of the `ub_camera` module.  **This is very much a work-in-progress**.  The code below was previously part of a Jupyter notebook, but it was tough to keep everyone's notebooks in sync.

---

### 1.  Import the package:
```
import ub_camera
```

### 2. Initialize your camera
There are 3 types of camera classes:
1. `CameraUSB` - This is for any camera that has a device path (like `/dev/video0`).  Examples include webcams, internal laptop cams, and even Raspberry Pi cameras.
2. `CameraROS` - This is for cameras that subscribe to compressedImage topic, including Gazebo simulations and the Clover drone (real hardware).
3. `CameraPi` - This is exclusive to Raspberry Pi cameras that use the `picamera` package.  This option is deprecated.

If you're unsure, chances are `CameraUSB` is the appropriate class for you.

``` 
# Initialize `CameraUSB` Class
paramDict = {'res_rows':480, 'res_cols':640, 'fps_target':30, 'outputPort': 8000}
apiPref   = None
device    = 0          # '/dev/video0'

camera = ub_camera.CameraUSB(paramDict = paramDict, device = device, apiPref = apiPref)
```
- **FIXME** -- Need to document the arguments in the `CameraUSB` class.

### 3.  Start the camera
```
camera.start()
```
- **Before you exit, make sure you stop your camera.**  See code below.

### 4. Stream the camera feed to be viewed in a browser
```
camera.startStream(port=8000)
```

- Visit https://localhost:8000/stream.mjpg
- NOTE:  You could combine the start and stream options into one command:
    ```
    camera.start(startStream=True, port=8000)
    ```
  
### 5.  When you're done with the camera, stop it:
```
camera.stop()
```

    
---  

## Aruco Tags
**NOTE** You will need to calibrate the camera if you want to be able to determine the distance from a tag.
- See `addCalibrate()` function    
    
### Start ArUco detection (choose the appropriate dictionary for your tags):    
```    
camera.addAruco('DICT_4X4_250', fps_target=20)
# camera.addAruco('DICT_APRILTAG_36h11', fps_target=20)
# camera.addAruco('DICT_APRILTAG_16h5', fps_target=20)
```

### Stop ArUco detection (make sure to use the same dictionary as above):
```
camera.aruco['DICT_4X4_250'].stop()
# camera.aruco['DICT_APRILTAG_36h11'].stop()
# camera.aruco['DICT_APRILTAG_16h5'].stop()
```

--- 

## Detect Barcodes

First, create a function that will be called each time a barcode is detected:
```
def barcodePost():
    try:
        if (len(camera.barcode['default'].deque) > 0):
            if (len(camera.barcode['default'].deque[0]['data']) > 0):
                print(camera.barcode['default'].deque)
    except Exception as e:
        pass
```

Next, start the barcode reader, pointing to the `barcodePost()` function:
```
camera.addBarcode(postFunction=barcodePost)
```

When you're done, stop the barcode reader:
```
camera.barcode['default'].stop()
```

--- 

## Face Detection

Start:
```
camera.addFaceDetect()
```

Stop:
```
camera.facedetect['default'].stop()
```




    
