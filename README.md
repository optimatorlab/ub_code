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

> **⚠️ STRONGLY RECOMMENDED: Use a Python Virtual Environment**
>
> Before installing dependencies, create and activate a virtual environment to isolate this package and its dependencies from your system Python:
>
> - Create a virtual environment (replace 'ub_code_env' with any name you prefer)
>     ```bash
>     python3 -m venv ub_code_env
>     ```
>
> - Linux/Mac -- Activate virtual environment:
>     ```bash
>     source ub_code_env/bin/activate
>     ```
>
> - Windows -- Activate virtual environment:
>     ```bash
>     ub_code_env\Scripts\activate
>     ```
>
> Your prompt should now show `(ub_code_env)`
>
>
> This prevents conflicts with other Python packages on your system and makes it easy to manage dependencies.

The installation order is critical because `ultralytics` may install `opencv-python`, but we need `opencv-contrib-python` for ArUco marker support.

**You can safely copy and paste this entire block:**

```bash
# Install numpy
pip install "numpy>=2.4.1"

# Install ultralytics (for YOLO object detection)
pip install "ultralytics>=8.4.7"

# Uninstall opencv-python (ultralytics may have installed this)
# The -y flag auto-confirms, so this is safe to copy-paste
pip uninstall -y opencv-python

# Uninstall opencv-contrib-python (in case it's already installed)
# You may see a warning if it's not installed - that's OK
pip uninstall -y opencv-contrib-python

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

---

## Checking for Updates

You can check if you have the latest version installed:

```python
import ub_camera

# Check version and get update notification
ub_camera.checkVersion()
# Output:
# Current version: 2025-02-19.0
# Latest version:  2025-02-20.1
# ⚠ Update available! Run: pip install --upgrade ub-code

# Or use silently for programmatic checks
current, latest, is_up_to_date = ub_camera.checkVersion(verbose=False)
if not is_up_to_date:
    print(f"Please update from {current} to {latest}")
```

The function compares your installed version against the latest version on the GitHub main branch.

---

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

This document describes some basic functionality of the `ub_camera` module.  

See [example Jupyter notebook](https://github.com/optimatorlab/ub_code/issues/5).

---

### 1.  Import the `ub_camera` and other useful packages:
```python
import ub_camera, ub_utils
import cv2
import numpy as np
```

### 2. Check `ub_camera` version:
```python
ub_camera.checkVersion()
```

### 3. Initialize your camera
There are 3 types of camera classes:
1. `CameraUSB` - This is for any camera that has a device path (like `/dev/video0`).  Examples include webcams, internal laptop cams, and even Raspberry Pi cameras.
2. `CameraROS` - This is for cameras that subscribe to compressedImage topic, including Gazebo simulations and the Clover drone (real hardware).
3. `CameraPi` - This is exclusive to Raspberry Pi cameras that use the `picamera` package.  This option is deprecated.

If you're unsure, chances are `CameraUSB` is the appropriate class for you.

```python 
# Specify port for streaming:
port = ub_utils.findOpenPort(8000, options=range(8000,8040))

# Define input device, image size, frames-per-second, etc:
device    = 0      # or 'https://10.83.17.66:8002/stream.mjpg' or '/dev/video0'
paramDict = {'res_rows':480, 'res_cols':640, 'fps_target':30, 'outputPort': port}
apiPref   = None   # on linux try cv2.CAP_V4L2 

# Initialize `CameraUSB` class, using default SSL certs
camera = ub_camera.CameraUSB(paramDict = paramDict, 
                             device = device, 
                             apiPref = apiPref,
                             showFPS=True)    # False --> Hide frames-per-second in video feed

# Start camera and stream:
camera.start(startStream=True, port=paramDict['outputPort'])

print(f'Visit https://localhost:{paramDict["outputPort"]}/stream.mjpg')
```
- **Before you exit, make sure you stop your camera.**  See code below.

  
### 4.  When you're done with the camera, stop it:
```python
camera.stopStream()
camera.stop()
```
    
---  

# Additional Tools

### Calibration

See [`calibration_example.ipynb`](https://github.com/optimatorlab/ub_code/issues/5) notebook for details.

```python
# This is copied from `calibration_example.ipynb`:
camera.intrinsics = { "640x480": {"fx": 613.9267755271052, "fy": 617.2876757419133, "cx": 326.06379688638367, "cy": 226.4726965669937, "dist": [-0.040671732389409375, 0.2205460570452358, -0.008313365917653356, 0.0025141234454979433, -0.32871689004906784]  } }
camera.intrinsics = camera._getIntrinsics()
camera.intrinsics
```
- **NOTE**: You might want to calibrate the camera for other resolutions, like `320x240`, too.

--- 

### Aruco Tags
- **NOTE**: You will need to calibrate the camera if you want to be able to determine the distance from a tag.

```python
# Specify the size of the ArUco tag in inches (or enter `None` if unknown)
TAG_SIZE_INCHES = 4.25   #  or None, or 4 + 3/16, etc

# Specify what type of ArUco tag you have:
ARUCO_DICTIONARY = 'DICT_APRILTAG_36h11'   # or 'DICT_4X4_250', or 'DICT_APRILTAG_16h5', etc
```
   
```python
# Define the "callback" function to be called on each ArUco detection:
def aruco_post_poses(argsDict):
    # This function gets called each time an aruco detection is run
    idName  = argsDict['idName']

    if (TAG_SIZE_INCHES is not None):
        # Adjust based on resolution
        res = f'{camera.res_cols}x{camera.res_rows}'
        cameraMatrix = camera.intrinsics[res]['matrix']
        dist = camera.intrinsics[res]['dist']

        # ********************************************
        # Specify the size of the marker, in [meters]
        # ********************************************
        ml = ub_utils.inches2meters(TAG_SIZE_INCHES)  

        objPoints = np.array([[-ml/2,  ml/2, 0], 
                              [ ml/2,  ml/2, 0], 
                              [ ml/2, -ml/2, 0], 
                              [-ml/2, -ml/2, 0]])
        
    corners = camera.aruco[idName].deque[0]['corners']
    for i in range(len(corners)):
        # centers give the center point, in pixels, of the tag.
        print(f"id: {camera.aruco[idName].deque[0]['ids'][i]}")
        print(f"\tcenter: {camera.aruco[idName].deque[0]['centers'][i]}")

        if (TAG_SIZE_INCHES is not None):
            '''
            NOTE:
            If you get an error like `Error in Aruco DICT_APRILTAG_36h11 thread: '640x480'`,
            that likely means you have not the camera calibration
            (or that you have calibrated your camera at a resolution other than '640x480'.
            '''
            (ret, rvecs, tvecs) = ub_utils.arucoFindPose(objPoints, corners[i], cameraMatrix, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            # print(f"{ret=}, {rvecs=}, {tvecs=}.")
            if (ret):
                # rvecs is the 3D rotation vector.  
                # I don't think it's human interpretable, so we won't print it here. 

                # tvecs in the x/y/z translation of the marker from the origin (camera).
                # It's in [meters], since we specified `ml` in [meters].
                # tvecs[0] (x) is the distance left (-) or right (+) from the camera.
                # tvecs[1] (y) is the distance above (-) or below (+) the camera.
                # tvecs[2] (z) is the distance away from the camera.
                print(f"\tdistance [inches]: x: {ub_utils.meters2inches(tvecs[0])}, y: {ub_utils.meters2inches(tvecs[1])}, z: {ub_utils.meters2inches(tvecs[2])}")
```

```python
# Start AruCo detection:
camera.addAruco(idName=ARUCO_DICTIONARY, 
                fps_target=5, 
                postFunction=aruco_post_poses, 
                postFunctionArgs={'idName': ARUCO_DICTIONARY}, 
                configOverrides={}, 
                ids_of_interest=None)  # default is None, or provide a list of IDs to track
```

**Run the next cell when you're ready to stop the ArUco detection:**
```python
camera.aruco[ARUCO_DICTIONARY].stop()
```
  
--- 

### Detect Barcodes and QR Codes

```python
# Create a function that will be called each time a barcode or QR code is detected:
def postBarcode(argsDict):
    # print(camera.barcode['default'].deque[0])
    for i in range(len(camera.barcode['default'].deque[0]['data'])): 
        print(f"""data: {camera.barcode['default'].deque[0]['data'][i]},  
                codeType: {camera.barcode['default'].deque[0]['codeTypes'][i]},
                quality: {camera.barcode['default'].deque[0]['qualities'][i]},
                corners: {camera.barcode['default'].deque[0]['corners'][i]}""")
```

```python
# Start the barcode reader, pointing to the `postBarcode()` function:
camera.addBarcode(fps_target=5,
                  postFunction=postBarcode)
```

**Run the next cell when you're ready to stop the barcode reader:**

```python
camera.barcode['default'].stop()
```


--- 

### Face Detection

```python
# Create a function that will be called each time a face is detected:
def postFaceDetect(argsDict):
    # print(camera.facedetect['default'].deque[0])
    for i in range(len(camera.facedetect['default'].deque[0]['confidence'])): 
        print(f"{i} - confidence: {camera.facedetect['default'].deque[0]['confidence'][i]}, corners: {camera.facedetect['default'].deque[0]['corners'][i]}")
```

```python
# Start the face detection
#
# Optional:  Specify where the OpenCV face detection models are saved.
# None --> Use default `cv2_dnn_models` included with ub_camera package.
modelPath = None
camera.addFaceDetect(fps_target=5,
                     postFunction=postFaceDetect, 
                     conf_threshold=0.7, 
                     dnn='caffe',    # 'caffe' (fp16) or 'pb' (8bit)
                     device='cpu', 
                     modelPath=modelPath)
```

**Run the next cell when you're ready to stop the face detection:**
```python
camera.facedetect['default'].stop()
```

--- 

### Ultralytics
The following options are documented:
- Detect
- Pose
- Oriented Bounding Box (obb)
- Segment (mask)
- Track (can be applied to `Detect`, `Pose`, and `Segment`)

The examples below use the YOLO 11 pre-trained models.  See https://docs.ultralytics.com/models/ for other options.

NOTE:  We should also explore the following:
- https://docs.ultralytics.com/models/rtdetr/#pretrained-models
- https://docs.ultralytics.com/models/sam-3/#training-data-scaling
- https://docs.ultralytics.com/models/mobile-sam/


#### Detect
```python
# Create a function that will be called each time an object is detected:
def postUltralyticsDetect(argsDict):
    idName = argsDict['idName']
    results = argsDict['results']
    
    for result in results:
        '''
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box    
        '''

        for i in range(0, len(result.boxes.cls)):
            # print(int(result.boxes.cls[i].item())
            # print(camera.ultralytics[idName].model.names[int(result.boxes.cls[i].item())])
            # print(result.boxes.conf[i].item(), result.boxes.xyxy[i].tolist())
            print(f'{result.names[int(result.boxes.cls[i].item())]} ({result.boxes.conf[i].item()}), {result.boxes.xyxy[i].tolist()}')
```

```python
# Start the object detection:
camera.addUltralytics(idName="detect", 
                      model_name="yolo11n.pt", 
                      conf_threshold=0.75, 
                      postFunction=postUltralyticsDetect)
```

```python
# Get list of objects that can be detected:
camera.ultralytics['detect'].model.names
```

```python
# Customize the annotation drawn on the video stream:
camera.ultralytics['detect'].drawBox   = True
camera.ultralytics['detect'].drawLabel = True
```
    
**Run the next cell when you're ready to stop the detection:**
```python    
camera.ultralytics['detect'].stop()    
```

#### Pose
```python
# Create a function that will be called each time a pose is detected:
def postUltralyticsPose(argsDict):
    idName = argsDict['idName']
    results = argsDict['results']
    
    '''
    `keypoints` should have 17 elements:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow, 9: Left Wrist, 10: Right Wrist,
    11: Left Hip, 12: Right Hip, 13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    '''
    
    for result in results:
        if (result.keypoints.has_visible):
            print(f'conf: {result.keypoints.conf.tolist()}, keypoints: {result.keypoints.xy.tolist()} \n')
```

```python
# Start the pose detection:
camera.addUltralytics(idName="pose", 
                      model_name="yolo11n-pose.pt", 
                      conf_threshold=0.75,  
                      postFunction=postUltralyticsPose, 
                      drawBox = False, drawLabel=True)
```
                      
```python
# Customize the annotation drawn on the video stream:
camera.ultralytics['pose'].drawBox   = False
camera.ultralytics['pose'].drawLabel = False
```
    
**Run the next cell when you're ready to stop the detection:**
```python    
camera.ultralytics['pose'].stop()
```

#### Oriented Bounding Boxes (OBB)
```python
# Create a function that will be called each time an oriented object is detected:
def postUltralyticsObb(argsDict):
    idName = argsDict['idName']
    results = argsDict['results']
    
    for result in results:
        if (result.obb):
            for i in range(0, len(result.obb.cls)):
                    print(f'{result.names[int(result.obb.cls[i].item())]} ({result.obb.conf[i].item()}), Center: {result.obb.xywhr[i][0:2].tolist()}')        
```
       
```python
# Start the obb detection:
camera.addUltralytics(idName="obb", 
                      model_name="yolo11n-obb.pt", 
                      conf_threshold=0.65,  
                      postFunction=postUltralyticsObb, 
                      drawBox = True, drawLabel=True)                    
```
 
```python
# Get list of objects that can be detected:
camera.ultralytics['obb'].model.names 
```

**Run the next cell when you're ready to stop the obb detection:**
```python
camera.ultralytics['obb'].stop()
```

#### Segmentation 
```python
# Create a function that will be called each time an object is detected:
def postUltralyticsSegment(argsDict):
    idName = argsDict['idName']
    results = argsDict['results']
    
    for result in results:
        for i in range(0, len(result.boxes.cls)):
            try:
                print(f'{result.names[int(result.boxes.cls[i].item())]} ({result.boxes.conf[i].item()}), {result.boxes.xyxy[i].tolist()}')   
            except Exception as e:
                print(f'Error: {e}')
```

```python
# Start the segmentation:
camera.addUltralytics(idName="segment", 
                      model_name="yolo11n-seg.pt", 
                      conf_threshold=0.65,  
                      postFunction=postUltralyticsSegment, 
                      drawBox = False, drawLabel=True, 
                      maskOutline = False)
```

```python
# Customize the annotation drawn on the video stream:
camera.ultralytics['segment'].maskOutline = True
```


```python
# Get list of objects that can be detected:
camera.ultralytics['segment'].model.names
```

**Run the next cell when you're ready to stop the segmentation:**
```python
camera.ultralytics['segment'].stop()
```

#### Tracking
```python
# Create a function that will be called each time an object is detected:
def postUltralyticsTrack(argsDict):
    idName = argsDict['idName']
    results = argsDict['results']
    
    # print(idName)   # "track"
    for result in results:
        '''
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        xywhn = result.boxes.xywhn  # normalized
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        xyxyn = result.boxes.xyxyn  # normalized
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        confs = result.boxes.conf  # confidence score of each box    
        '''
        for i in range(0, len(result.boxes.cls)):
            try:
                print(f'ID: {result.boxes.id[i].item()} - {result.names[int(result.boxes.cls[i].item())]} ({result.boxes.conf[i].item()}), {result.boxes.xyxy[i].tolist()}')                
            except Exception as e:
                print(f'Error: {e}')
```

```python                
# Tracking can be done with detect, pose, or segment models.
# Choose one of the following
model_name = "yolo11n.pt"          # detect
# model_name = "yolo11n-pose.pt"   # pose
# model_name = "yolo11n-seg.pt"    # segment
```

```python
# Start tracking:
camera.addUltralytics(idName="track", 
                      model_name=model_name, 
                      conf_threshold=0.65,  
                      postFunction=postUltralyticsTrack, 
                      drawBox = False, drawLabel=True) 
```
                      
```python
# Customize the annotation drawn on the video stream:                      
camera.ultralytics['track'].drawBox = False
camera.ultralytics['track'].drawLabel = True
```

**Run the next cell when you're ready to stop the tracking:**
```python
camera.ultralytics['track'].stop()               
```

---

### Timelapse
Take photos at regular intervals, saving them to a directory on your computer.

```python
'''
outputDir: Folder where the photos will be saved.  Use relative directory or absolute path.
secBetwPhotos: How many seconds between photo captures.
timeLimitSec: Keep capturing photos for this many seconds.  `None` --> No limit.
delayStartSec: How many seconds to wait before taking the first picture.
postPostFunction: Function to call when the timelapse is finished. 
'''

camera.addTimelapse(outputDir        = 'timelapse_photos',    
                    secBetwPhotos    = 3, 
                    timeLimitSec     = None, 
                    delayStartSec    = 0, 
                    postPostFunction = None)
```

**Run the next cell when you're ready to stop the timelapse:**
```python
camera.timelapse['default'].stop()
```                    

--- 

### Video from Pics
- TBD.  First, run timelapse to save photos to a directory, then process the photos in that directory into an `.mpeg` video.
    
### Region of Interest (ROI)
- Deprecated.  This functionality would (poorly) track a selected object.  The Ultralytics tracking is better (although it's limited to trained objects).
                                                                                                    
