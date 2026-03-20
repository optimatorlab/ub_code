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
pip install "numpy>=1.21.0"

# Install ultralytics (for YOLO object detection)
pip install "ultralytics>=8.3.0"

# Uninstall opencv-python (ultralytics may have installed this)
# The -y flag auto-confirms, so this is safe to copy-paste
pip uninstall -y opencv-python

# Uninstall opencv-contrib-python (in case it's already installed)
# You may see a warning if it's not installed - that's OK
pip uninstall -y opencv-contrib-python

# Install opencv-contrib-python (required for ArUco markers, face detection, etc.)
pip install "opencv-contrib-python>=4.10.0"

# Install pyzbar (for barcode/QR code detection)
pip install "pyzbar>=0.1.9"

# Install websockets (for websocket streaming)
pip install "websockets>=12.0"

# Install items for WebRTC streaming
pip install "aiortc>=1.9.0"
pip install "aiohttp>=3.9.0"
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

**With WebSocket streaming support (optional):**
```bash
pip install -e ".[websocket]"
```

**With WebRTC streaming support (optional):**
```bash
pip install -e ".[webrtc]"
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
```

```python
# Or use silently for programmatic checks
current, latest, is_up_to_date = ub_camera.checkVersion(verbose=False)
if not is_up_to_date:
    print(f"Please update from {current} to {latest}")
```

The function compares your installed version against the latest version on the GitHub main branch.

---

## Streaming Protocols

`ub_camera` supports three streaming protocols. All use TLS so they can be
embedded in `https://` pages without mixed-content errors.

| Protocol | Extra install | Typical latency | Browser endpoint | Multi-client |
|---|---|---|---|---|
| **MJPEG** (default) | None | 200–500 ms | `https://host:PORT/stream.mjpg` | Yes |
| **WebSocket + JPEG** | `ub-code[websocket]` | 100–300 ms | See snippet below | Yes |
| **WebRTC** | `ub-code[webrtc]` | 50–150 ms | `https://host:PORT/webrtc` | Yes |

### MJPEG (default)

No extra dependencies. All existing code continues to work unchanged.

```python
camera.startStream(port=8000)                    # default
camera.startStream(port=8000, protocol='mjpeg')  # explicit
# Visit https://host:8000/stream.mjpg
```

### WebSocket + JPEG

```bash
pip install "ub-code[websocket]"
```

```python
camera.startStream(port=8001, protocol='websocket')
```

Embed in a web page:

```html
<canvas id="cam" width="640" height="480"></canvas>
<script>
  const canvas = document.getElementById('cam');
  const ctx    = canvas.getContext('2d');
  const ws     = new WebSocket('wss://camera-host:8001');
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (e) => {
    const blob = new Blob([e.data], { type: 'image/jpeg' });
    createImageBitmap(blob).then(bmp => ctx.drawImage(bmp, 0, 0));
  };
</script>
```

### WebRTC

```bash
pip install "ub-code[webrtc]"
```

```python
camera.startStream(port=8002, protocol='webrtc')
# Built-in viewer: https://host:8002/webrtc
```

`GET /webrtc` serves a self-contained HTML+JS page — open it directly in a
browser with no additional setup. The media stream uses WebRTC DTLS and never
triggers a TLS certificate warning, regardless of whether the signaling
endpoint uses a self-signed or trusted cert.

For integration into your own web page, use `signalingMode='json'`:

```python
camera.startStream(port=8002, protocol='webrtc', signalingMode='json')
# GET  /webrtc  →  {"offerUrl": "/offer"}
# POST /offer   accepts {sdp, type}, returns {sdp, type}
```

### Switching protocols

Only one protocol is active at a time per camera. To switch without stopping
first, pass `force=True`:

```python
camera.startStream(port=8002, protocol='webrtc', force=True)
```

---

## Using Custom SSL Certificates

The package includes self-signed SSL certificates for HTTPS/WSS streaming
(useful for development and testing). All three streaming protocols use the
same certificate. To use your own certificates instead:

```python
camera = ub_camera.CameraUSB(
    paramDict={'res_rows': 480, 'res_cols': 640, 'fps_target': 30},
    sslPath='/path/to/your/ssl/directory'
)
```

Your SSL directory should contain:
- `ca.crt` — SSL certificate file
- `ca.key` — SSL private key file

If you don't specify `sslPath`, the package uses the bundled certificates in
`ub_camera/ssl/`. For deployments where users should not see a browser
security warning, replace the bundled cert with a trusted one (e.g., from a
university subdomain or a Let's Encrypt reverse proxy). No code changes are
needed — only the certificate files change.

---

## Reverse Proxy Deployment (Zero Browser Warnings)

For multi-user deployments where students access camera streams from personal
machines, a reverse proxy with a trusted certificate eliminates the self-signed
cert browser warning entirely — with no client-side setup required.

**Key principle:** For WebRTC, the proxy handles only the small signaling
messages (`GET /webrtc`, `POST /offer`). The actual video stream flows directly
from the camera device to the browser via WebRTC DTLS — it never touches the
proxy. Latency is unaffected.

### Architecture

```
Student browser
    │  HTTPS signaling (small JSON messages only)
    ▼
Reverse proxy   ←── your domain, trusted cert, public/campus network
    │  forwards to
    ▼
Camera device   ←── local network, self-signed cert or plain HTTP internally
    │  WebRTC media (UDP, direct)
    └──────────────────────────────────────► Student browser
```

### Caddy (recommended — automatic HTTPS)

[Caddy](https://caddyserver.com) automatically provisions and renews
Let's Encrypt certificates. Install it on any internet-accessible server
(a university VM, cloud instance, etc.).

**`Caddyfile`** — one block per camera device:

```
cameras.yourdomain.com {
    # Route each camera to its own subdirectory
    handle /camera1/* {
        uri strip_prefix /camera1
        reverse_proxy camera1-hostname:8002
    }
    handle /camera2/* {
        uri strip_prefix /camera2
        reverse_proxy camera2-hostname:8002
    }
}
```

Start Caddy:
```bash
caddy run --config Caddyfile
```

Students access the built-in WebRTC viewer at:
```
https://cameras.yourdomain.com/camera1/webrtc
```

The camera devices themselves need no configuration change. Caddy terminates
TLS on behalf of the camera; internally it can proxy to either HTTP or HTTPS
(the camera's self-signed cert only needs to be trusted by the proxy, not
by the student's browser).

To allow Caddy to proxy to the camera's self-signed HTTPS endpoint:
```
handle /camera1/* {
    uri strip_prefix /camera1
    reverse_proxy camera1-hostname:8002 {
        transport http {
            tls_insecure_skip_verify   # proxy trusts camera's self-signed cert
        }
    }
}
```

Alternatively, run the camera on plain HTTP internally (no SSL) and let
Caddy provide TLS at the edge — students still get `https://`, and the
internal hop stays on a trusted LAN:

```python
# Camera side: serve without SSL (LAN-only, behind the proxy)
# Not yet supported — use sslPath with a self-signed cert for now,
# and set tls_insecure_skip_verify on the Caddy block above.
```

### Apache

Required modules (enable with `a2enmod` on Debian/Ubuntu):

```bash
sudo a2enmod ssl proxy proxy_http proxy_wstunnel rewrite
sudo systemctl reload apache2
```

**VirtualHost config** (`/etc/apache2/sites-available/cameras.conf`):

```apache
<VirtualHost *:443>
    ServerName cameras.yourdomain.com

    SSLEngine               On
    SSLCertificateFile      /etc/ssl/certs/yourdomain.crt
    SSLCertificateKeyFile   /etc/ssl/private/yourdomain.key
    # SSLCertificateChainFile /etc/ssl/certs/chain.crt   # if required by your CA

    # Allow proxying to the camera's self-signed HTTPS cert
    SSLProxyEngine          On
    SSLProxyVerify          none
    SSLProxyCheckPeerCN     Off
    SSLProxyCheckPeerName   Off

    # ---------------------------------------------------------------
    # Camera 1 — WebRTC (signaling only; media flows direct via UDP)
    # ---------------------------------------------------------------
    ProxyPass        /camera1/  https://camera1-hostname:8002/
    ProxyPassReverse /camera1/  https://camera1-hostname:8002/

    # ---------------------------------------------------------------
    # Camera 1 — WebSocket streaming (wss://)
    # ---------------------------------------------------------------
    RewriteEngine On
    RewriteCond   %{HTTP:Upgrade} websocket [NC]
    RewriteCond   %{HTTP:Connection} upgrade [NC]
    RewriteRule   ^/camera1-ws/(.*)  wss://camera1-hostname:8001/$1  [P,L]

    ProxyPass        /camera1-ws/  wss://camera1-hostname:8001/
    ProxyPassReverse /camera1-ws/  wss://camera1-hostname:8001/

    # Repeat the above blocks for additional cameras
    # (camera2 → port 8002, etc.)
</VirtualHost>

# Redirect plain HTTP to HTTPS
<VirtualHost *:80>
    ServerName cameras.yourdomain.com
    Redirect permanent / https://cameras.yourdomain.com/
</VirtualHost>
```

Enable and reload:
```bash
sudo a2ensite cameras.conf
sudo systemctl reload apache2
```

Students access the built-in WebRTC viewer at:
```
https://cameras.yourdomain.com/camera1/webrtc
```

> **Note on WebSocket path:** Because Apache proxies `/camera1-ws/` to the
> camera's WebSocket port (8001), the browser's `WebSocket()` URL must use
> that path prefix:
> ```js
> const ws = new WebSocket('wss://cameras.yourdomain.com/camera1-ws/');
> ```
> Adjust accordingly if you use a different URL scheme.

### Nginx

For environments where Nginx is already deployed:

```nginx
server {
    listen 443 ssl;
    server_name cameras.yourdomain.com;

    ssl_certificate     /etc/ssl/certs/yourdomain.crt;
    ssl_certificate_key /etc/ssl/private/yourdomain.key;

    location /camera1/ {
        rewrite ^/camera1/(.*) /$1 break;
        proxy_pass https://camera1-hostname:8002;
        proxy_ssl_verify off;          # camera uses self-signed cert

        # Required for WebSocket upgrade (websocket protocol)
        proxy_http_version  1.1;
        proxy_set_header    Upgrade    $http_upgrade;
        proxy_set_header    Connection "upgrade";
    }
}
```

### Access control with the proxy

The camera's `ipAllowlist` / `ipBlocklist` will see the **proxy's IP**, not
the student's IP, when traffic is forwarded. If per-student IP filtering is
needed, either:
- Apply access control at the proxy level (Apache `Require ip`, Caddy
  `basicauth`, Nginx `allow`/`deny`), or
- Pass the student's real IP via `X-Forwarded-For` and update the camera
  access-control logic to read that header (not currently implemented).

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

# Start camera and stream (MJPEG default):
camera.start(startStream=True, port=paramDict['outputPort'])

print(f'Visit https://localhost:{paramDict["outputPort"]}/stream.mjpg')

# Or start with a different protocol after camera.start():
# camera.startStream(port=paramDict['outputPort'], protocol='websocket')
# camera.startStream(port=paramDict['outputPort'], protocol='webrtc')
# print(f'Visit https://localhost:{paramDict["outputPort"]}/webrtc')

print("When you're done, be sure to stop the camera: camera.stop()")
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

### Circle and Text Overlays

You can add circle and text overlays to the video stream. Both return a `(decorationID, params)` tuple. The `params` dict is mutable — update its values to change the overlay dynamically each frame.

#### Circle
```python
# Add a circle at (center_x, center_y) with radius 50
cid, circle_params = camera.addCircle(center=(320, 240), radius=50, thickness=3, color=(150, 25, 25))
```

```python
# Move the circle dynamically:
circle_params['center'] = (400, 300)
circle_params['radius'] = 75
circle_params['color'] = (0, 255, 0)
```

```python
# Remove the circle:
camera.removeDecoration(cid)
```

#### Text
```python
# Add text at position (x, y)
tid, text_params = camera.addText(text="Hello", position=(100, 100), fontScale=0.7, thickness=2, color=(255, 255, 255))
```

```python
# Update the text dynamically:
text_params['text'] = "World"
text_params['position'] = (200, 200)
text_params['color'] = (0, 0, 255)
```

```python
# Remove the text:
camera.removeDecoration(tid)
```

---

### Video from Pics
- TBD.  First, run timelapse to save photos to a directory, then process the photos in that directory into an `.mpeg` video.
    
### Region of Interest (ROI)
- Deprecated.  This functionality would (poorly) track a selected object.  The Ultralytics tracking is better (although it's limited to trained objects).
                                                                                                    
