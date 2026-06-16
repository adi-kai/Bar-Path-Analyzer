# Bar-Path-Analyzer


# What is it?
A Python script that takes a weightlifting video as input and
outputs an annotated video with:


* Color-coded bar path trail (blue=slow → red=fast)
* Peak height annotation (dot + label box, floor-relative)
* Peak velocity annotation (dot + label box)
* MediaPipe Pose skeleton overlay on every frame
* Speed legend at bottom left
* Title bar at top

# Working enviroment:
* OS: Windows (PowerShell)
* Python: 3.11.9  (MUST use this, not 3.14)
* MediaPipe: 0.10.9  (MUST use this specific version)
* OpenCV: latest (opencv-python)
* NumPy: latest

Why 3.11 not 3.14: 
MediaPipe does not support Python 3.14.

# How to run?
1st. Install deps (one time)
py -3.11 -m pip install opencv-python mediapipe==0.10.9 numpy

2nd. Run
py -3.11 bar_path_analyzer.py 
--input "yourfile.mp4" 
--output result.avi 
--bar_color (red | blue | black | green | yellow | auto)

Example: py -3.11 bar_path_analyzer.py --input "snatch.mp4" --output result.avi --bar_color green 

# How does this work?
PASS 1 — Detection
* Reads every frame
* ROI mask: center band (w//6 to 5w//6, full height)
Prevents locking onto racks and far-background objects
* Converts to HSV, applies color range mask
* Morphological cleanup (close + open, 7x7 ellipse kernel)
* Contour filtering:
* Minimum area: 400px²
* Circularity >= 0.55 (partially occluded plates ~0.45-0.55)
* Diameter: 4%–35% of frame width
* Max-jump filter: rejects detections > max_jump_px from
previous frame (false positives can't teleport)
* Post-gap check: after a None streak, new detection must be within 3x max_jump of last known position
* Stores (cx, cy) or None per frame

PASS 2 — Render
* Works on frame.copy() — never mutates stored frame buffer
* Runs MediaPipe Pose → draws skeleton
* Draws color-coded trail over rolling window (fps * 10 frames)
* Velocity colormap: blue→green→cyan→yellow→red (BGR values)
* Max velocity = 95th percentile of nonzero velocities (prevents single outlier collapsing entire colormap)
* Peak height: floor_y - peak_y / PX_PER_METER (floor-relative)
* Writes to VideoWriter (XVID/.avi recommended on Windows)

# COLOR RANGES (HSV)
red (bright):   H 0-10 / 170-180,  S 120-255, V 70-255
red (maroon):   H 0-15 / 165-180,  S 80-180,  V 30-130
blue:           H 100-130,          S 150-255, V 50-255
black:          H 0-180,            S 0-255,   V 20-130
green:          H 35-90,            S 40-255,  V 40-255
yellow:         H 20-35,            S 100-255, V 100-255

# VIDEO RECORDING GUIDELINES (CRITICAL FOR DETECTION)
Best plate colors:    RED or BLUE (high saturation, unique HSV)
Avoid:                Black or maroon plates with dark clothing
Camera distance:      Far enough that plates stay fully in frame
at lockout overhead
Background:           Solid light color (grey/white wall = good)
Angle:                Side profile (already correct)
Lighting:             Even, no strong shadows across the plates
Fallback if dark plates: wrap bright colored tape around collar

# Next steps
* Add --athlete_height flag so the script can calculate real
meters instead of the current frame-height heuristic. Would
use MediaPipe's detected ankle-to-hip distance as a pixel
ruler, then scale using the athlete's known height.
*Export a velocity-over-time graph as a PNG alongside the
video — shows the bar's speed curve across the whole lift
so you can see exactly when peak pull happens.
*Add --start and --end flags to clip the video by timestamp
so you don't have to manually trim before running the script.
  Example: --start 0:03 --end 0:07
* Wrap the whole thing in a Streamlit web UI

  
